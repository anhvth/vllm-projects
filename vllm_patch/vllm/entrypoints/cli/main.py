# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)


def main():
    import vllm.entrypoints.cli.benchmark.main as benchmark_main
    import vllm.entrypoints.cli.collect_env as collect_env
    import vllm.entrypoints.cli.launch as launch
    import vllm.entrypoints.cli.openai as openai
    import vllm.entrypoints.cli.run_batch as run_batch
    import vllm.entrypoints.cli.serve as serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        openai,
        serve,
        launch,
        benchmark_main,
        collect_env,
        run_batch,
    ]

    cli_env_setup()

    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" in sys.argv:
        # NOTE: Check the spec instead of importing directly here, since things could
        # fail with ImportError due to mismatched versions if things are moved around.
        spec = find_spec("vllm_omni")
        if spec is None:
            logger.error(
                "--omni flag requires a valid instance of vllm-omni to be installed."
            )
            sys.exit(1)

        from vllm_omni.entrypoints.cli.main import main as omni_main  # pyright: ignore[reportMissingImports]

        logger.info("Delegating entrypoint handling to vllm-omni")
        omni_main()
    else:
        # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
        if len(sys.argv) > 1 and sys.argv[1] == "bench":
            logger.debug(
                "Bench command detected, must ensure current platform is not "
                "UnspecifiedPlatform to avoid device type inference error"
            )
            import vllm.platforms as platforms

            if platforms.current_platform.is_unspecified():
                from vllm.platforms.cpu import CpuPlatform

                platforms.current_platform = CpuPlatform()
                logger.info(
                    "Unspecified platform detected, switching to CPU Platform instead."
                )

        parser = FlexibleArgumentParser(
            description="vLLM CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm"),
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        subparser = getattr(args, "subparser", None)
        if subparser in cmds:
            cmds[subparser].validate(args)

        dispatch_function = getattr(args, "dispatch_function", None)
        if dispatch_function is not None:
            dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
