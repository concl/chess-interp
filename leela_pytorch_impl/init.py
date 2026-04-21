import gzip
import logging
import os
import sys
from typing import Optional

import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.convert.leela_to_jax import (
    LeelaImportOptions,
    fix_older_weights_file,
    leela_to_jax,
)
from lczero_training.convert.leela_to_modelconfig import leela_to_modelconfig
from lczero_training.training.state import TrainingState
from proto import hlo_pb2, net_pb2
from proto.model_config_pb2 import ModelConfig
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def _load_lc0_model_state(
    path: str,
    expected_config: ModelConfig,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
    ignore_config_mismatch: bool = False,
) -> tuple[nnx.State, int]:
    """Load lc0 weights, validate config, return (model_state, training_steps)."""
    lc0_weights = net_pb2.Net()
    with gzip.open(path, "rb") as f:
        lc0_weights.ParseFromString(f.read())
    fix_older_weights_file(lc0_weights)

    leela_config = leela_to_modelconfig(
        lc0_weights, hlo_pb2.XlaShapeProto.F32, compute_dtype
    )
    if leela_config != expected_config:
        if ignore_config_mismatch:
            logger.warning(
                "The provided lczero model configuration "
                "differs from the one in the config file (ignored)."
            )
        else:
            logger.error(
                "The provided lczero model configuration "
                "differs from the one in the config file."
            )
            logger.error(f"Config file model config: {expected_config}")
            logger.error(f"Leela model config: {leela_config}")
            sys.exit(1)

    import_options = LeelaImportOptions(
        weights_dtype=hlo_pb2.XlaShapeProto.F32, compute_dtype=compute_dtype
    )
    model_state = leela_to_jax(lc0_weights, import_options)
    return model_state, lc0_weights.training_params.training_steps
