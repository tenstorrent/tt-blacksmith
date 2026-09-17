# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
def disable_forge_logger():
    from loguru import logger

    logger.disable("")
