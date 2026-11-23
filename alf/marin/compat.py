"""Runtime shims that let Marin's grug MoE hero code run unmodified on the hardware and library versions used here.

None of them changes the model math, and none of them edits the shared Marin checkout or JAX; each works around one
thing the upstream code does not cover:

- Pallas: JAX's Pallas Triton lowering identifies the accelerator by its device-kind name and looks that name up in a
  fixed table. A GPU whose kind is absent from the table cannot run the lowering even when its compute capability is
  one the table already supports, so the kernels fail on a device that is otherwise fine. The lowering exposes
  `registry` for exactly this case: registering the missing kind with its architecture and compute capability is
  enough, and that is all the registration below does.
- Newton-Schulz: on a mesh whose axes all have size 1 (a single GPU), MuonH's stacked Newton-Schulz keeps the
  parameter's named `model` / `data` specs on the contracted dimensions and JAX's explicit-sharding check rejects the
  einsum. Marin never runs that path (it reshards over the batch axes first). Replicating the stack before the same
  per-matrix iteration is a no-op for the data on one device.
- Checkpoints: with TensorStore 0.1.84 on the `file` kvstore, a write that may reference its source indefinitely
  (levanter's `can_reference_source_data_indefinitely=True`) commits successfully yet stores zeros for some 4-D
  arrays, e.g. the d1024 expert banks [layers, 384, 512, 512] fp32 and their MuonH momentum; 2-D / 3-D arrays of the
  same size are stored intact, and so is the same write with False. A resume then restores zero experts, which
  MuonH's norm-preserving step keeps at zero for the rest of the run, with no error anywhere.
"""
from __future__ import annotations

# The device kinds registered below, as JAX reports them (the lowering compares the name verbatim). The first family
# is sm_90, the architecture the training runs used.
SM90_KINDS = ("NVIDIA H800", "NVIDIA H800 PCIe")
# An sm_80 family is registered the same way, so the evaluation kernels run on those devices too. Readings from
# different architectures are not interchangeable: the same checkpoint on the same questions gave per-question NLLs
# about 0.05 nats apart on two of them, so a single comparison must stay on one architecture.
SM80_KINDS = ("NVIDIA A800-SXM4-80GB", "NVIDIA A800 80GB PCIe")


def patch_checkpoint_write_copies_source() -> None:
    """Rebuild levanter's `_serialize_arrays` with `can_reference_source_data_indefinitely=False` (one call site)."""
    import inspect
    import textwrap

    import levanter.tensorstore_serialization as tss

    if getattr(tss._serialize_arrays, "_alf_copies_source", False):
        return
    src = textwrap.dedent(inspect.getsource(tss._serialize_arrays))
    old = "can_reference_source_data_indefinitely=True"
    if src.count(old) != 1:
        raise RuntimeError(f"levanter _serialize_arrays changed: expected one `{old}`, found {src.count(old)}")
    exec(compile(src.replace(old, "can_reference_source_data_indefinitely=False"), tss.__file__, "exec"), tss.__dict__)
    tss._serialize_arrays._alf_copies_source = True


def register_gpu_kinds_for_pallas() -> None:
    """Add the device kinds above to the Pallas Triton lowering's registry, each with its compute capability."""
    from jax._src.pallas.triton import gpu_info

    for kind in SM90_KINDS:
        gpu_info.registry.setdefault(
            kind, lambda: gpu_info.GpuInfo(gpu_version=None, arch_name="9.0", compute_capability=90)
        )
    for kind in SM80_KINDS:
        gpu_info.registry.setdefault(
            kind, lambda: gpu_info.GpuInfo(gpu_version=None, arch_name="8.0", compute_capability=80)
        )


def patch_single_device_newton_schulz() -> None:
    """Run MuonH's stacked Newton-Schulz on a replicated stack when the mesh holds a single device."""
    import jax
    from jax.sharding import PartitionSpec as P

    from experiments.grug.moe_hero_ep import grugmuon_hero as gm

    def single_device(mesh) -> bool:
        return not mesh.empty and not gm._intra_rack_axes(mesh)

    def local(steps, eps, coefficient_type):
        return lambda m: gm._zeropower_via_newtonschulz_local(m, steps, eps, coefficient_type)

    orig_stack = gm._newtonschulz_padded_stack_sharded
    orig_4d = gm._newtonschulz_4d_distributed

    def stack(X, steps=5, eps=1e-7, coefficient_type="quintic", *, target_sharding=None):
        if not single_device(jax.sharding.get_abstract_mesh()):
            return orig_stack(X, steps, eps, coefficient_type, target_sharding=target_sharding)
        out = jax.vmap(local(steps, eps, coefficient_type))(jax.sharding.reshard(X, P(None, None, None)))
        return jax.sharding.reshard(out, target_sharding) if target_sharding is not None else out

    def four_d(path, x, steps, eps, coefficient_type, use_syrk, target_sharding=None):
        if not single_device(jax.sharding.get_abstract_mesh()):
            return orig_4d(path, x, steps, eps, coefficient_type, use_syrk, target_sharding=target_sharding)
        out = jax.vmap(jax.vmap(local(steps, eps, coefficient_type)))(jax.sharding.reshard(x, P(None, None, None, None)))
        return jax.sharding.reshard(out, target_sharding) if target_sharding is not None else out

    gm._newtonschulz_padded_stack_sharded = stack
    gm._newtonschulz_4d_distributed = four_d
