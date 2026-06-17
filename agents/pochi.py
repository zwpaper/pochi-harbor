import os
import re
import json
import shlex
import hashlib

from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from pydantic import BaseModel
from pathlib import Path, PurePosixPath

from harbor.models.trajectories import (
    Trajectory,
    Step,
    ToolCall,
    Observation,
    ObservationResult,
    FinalMetrics,
    Agent,
)
from harbor.models.trial.paths import EnvironmentPaths


class InitialStateError(Exception):
    pass

class NoApiKeyError(Exception):
    pass


class ExecInput(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: int | None = None


class Pochi(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "pochi"

    @property
    def _trajectory_path(self) -> PurePosixPath:
        return PurePosixPath(EnvironmentPaths.agent_dir / "trajectory.json")

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.pochi/bin:$PATH"; export POCHI_LOG=info pochi; pochi --version'

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    async def install(self, environment: BaseEnvironment) -> None:
        # Install system packages (root)
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl ripgrep",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        # Install pochi (as default user). Pin a known-good CLI release when
        # the trial config doesn't specify one — getpochi.com/install.sh's
        # `get_latest_version` mis-parses the GitHub releases API from inside
        # some containers (returns the API URL as the "version" → 404).
        # Keep `bash -x` so any future install failure shows the resolved
        # version and download URL in the trace.
        version_spec = f"pochi-{self._version}" if self._version else "pochi-v0.6.14"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"curl -fsSL https://getpochi.com/install.sh | bash -x -s -- {version_spec} && "
                "mkdir -p /logs/agent/pochi && "
                "~/.pochi/bin/pochi --version"
            ),
        )
        # Symlink pochi to /usr/local/bin (root)
        await self.exec_as_root(
            environment,
            command="ln -sf ~/.pochi/bin/pochi /usr/local/bin/pochi",
        )

    def _build_register_skills_command(self) -> str | None:
        """Return a shell command that copies skills to Pochi's skills directory."""
        if not self.skills_dir:
            return None
        return (
            f"mkdir -p $HOME/.agents/skills && "
            f"cp -r {shlex.quote(self.skills_dir)}/* "
            f"$HOME/.agents/skills/ 2>/dev/null || true"
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        """Return a shell command that writes MCP config to ~/.pochi/config.jsonc."""
        if not self.mcp_servers:
            return None
        mcp_entries: list[str] = []
        for server in self.mcp_servers:
            if server.transport == "stdio":
                cmd_parts = [server.command] + server.args if server.command else []
                mcp_entries.append(
                    f'  {json.dumps(server.name)}: {{"command": {json.dumps(shlex.join(cmd_parts))}}}'
                )
            else:
                mcp_entries.append(
                    f'  {json.dumps(server.name)}: {{"url": {json.dumps(server.url)}}}'
                )
        mcp_block = "{\n" + ",\n".join(mcp_entries) + "\n}"
        escaped = shlex.quote(mcp_block)
        return (
            f'python3 -c "'
            f"import json, pathlib; "
            f"p = pathlib.Path('$HOME/.pochi/config.jsonc'); "
            f"cfg = json.loads(p.read_text()) if p.exists() else {{}}; "
            f"cfg.setdefault('mcpServers', {{}}).update(json.loads({escaped})); "
            f'p.write_text(json.dumps(cfg, indent=2))"'
        )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        model = self.model_name if self.model_name else "google/gemini-3-flash"

        config_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "DEEPINFRA_API_KEY": os.environ.get("DEEPINFRA_API_KEY", ""),
        }

        config_json = """{
  "providers": {
    "deepinfra": {
      "kind": "openai",
      "baseURL": "https://api.deepinfra.com/v1/openai",
      "apiKey": "DEEPINFRA_API_KEY",
      "models": {
        "zai-org/GLM-4.7-Flash": {
          "name": "glm-4.7-flash"
        },
        "moonshotai/Kimi-K2.6": {
          "name": "kimi-K2.6"
        },
        "deepseek-ai/DeepSeek-V4-Pro": {
          "name": "DeepSeek-V4-Pro"
        }
      }
    },
    "openai": {
      "kind": "openai",
      "baseURL": "https://api.openai.com/v1",
      "apiKey": "OPENAI_API_KEY",
      "models": {
        "gpt-5.4": {
          "name": "gpt-5.4"
        }
      }
    }
  }
}"""

        # Write config and set up environment
        setup_command = (
            "mkdir -p ~/.pochi && "
            "cat << 'EOF' | sed "
            '-e "s/OPENAI_API_KEY/${OPENAI_API_KEY}/g" '
            '-e "s/DEEPINFRA_API_KEY/${DEEPINFRA_API_KEY}/g" '
            "> ~/.pochi/config.jsonc\n"
            f"{config_json}\n"
            "EOF"
        )

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        await self.exec_as_agent(
            environment,
            command=setup_command,
            env=config_env,
        )

        pochi_api_key = os.environ.get("POCHI_API_KEY", "")
        if not pochi_api_key:
            raise NoApiKeyError("POCHI_API_KEY environment variable is not set.")

        eval_env = {
            "POCHI_LOG": "debug",
            "POCHI_API_KEY": pochi_api_key,
        }
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "if [ -f /bootstrap/test_initial_state.py ]; then "
                    "pytest -s --log-cli-level=DEBUG /bootstrap/test_initial_state.py "
                    "> >(tee /logs/agent/pochi/initial-test-stdout.txt) "
                    "2> >(tee /logs/agent/pochi/initial-test-stderr.txt >&2); "
                    "fi"
                ),
                env=eval_env,
            )
        except Exception as e:
            raise InitialStateError("Initial state test failed") from e

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "pochi "
                    f"--model {model} "
                    "--max-steps 200 "
                    "--max-retries 10 "
                    "--blobs-dir /logs/agent/pochi/blobs "
                    "--experimental-stream-trajectory /logs/agent/pochi/trajectory.jsonl "
                    "> >(tee /logs/agent/pochi/stdout.txt) "
                    "2> >(tee /logs/agent/pochi/stderr.txt >&2) "
                    "<<'EOF'\n"
                    f"{instruction}\n"
                    "EOF"
                ),
                env=eval_env,
            )
        finally:
            # cleanup - best effort
            try:
                await self.exec_as_agent(
                    environment,
                    command="rm -f ~/.pochi/config.jsonc",
                )
            except Exception:
                pass

    def _convert_pochi_to_atif(self, log_lines: list[str]) -> Trajectory | None:
        """Convert Pochi trajectory format to ATIF format."""
        if not log_lines:
            return None

        steps: list[Step] = []
        step_id = 1

        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        session_id = "unknown"
        total_total_tokens = 0
        total_system_tokens = 0
        total_tools_tokens = 0

        for line in log_lines:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_id = msg.get("id", "unknown")
            if session_id == "unknown" and msg_id != "unknown":
                session_id = msg_id

            role = msg.get("role")

            if role == "user":
                text_parts = [
                    p.get("text", "")
                    for p in msg.get("parts", [])
                    if p.get("type") == "text"
                ]
                content = "\n".join(text_parts)
                steps.append(
                    Step(
                        step_id=step_id,
                        source="user",
                        message=content,
                    )
                )
                step_id += 1
            elif role == "assistant":
                metadata = msg.get("metadata", {})

                total_tokens = metadata.get("totalTokens", 0)
                total_total_tokens += total_tokens
                system_tokens = metadata.get("systemPromptTokens", 0)
                total_system_tokens += system_tokens
                tools_tokens = metadata.get("toolsTokens", 0)
                total_tools_tokens += tools_tokens

                prompt_tokens = system_tokens + tools_tokens
                completion_tokens = max(total_tokens - prompt_tokens, 0)

                total_input_tokens += prompt_tokens
                total_output_tokens += completion_tokens

                parts = msg.get("parts", [])

                current_reasoning = []
                current_tools = []
                current_observations = []
                current_text = []

                def _flush_step():
                    nonlocal step_id, current_reasoning
                    nonlocal current_tools, current_observations
                    nonlocal current_text
                    if (
                        current_tools
                        or current_reasoning
                        or current_observations
                        or current_text
                    ):
                        reasoning_content = (
                            "\n".join(current_reasoning) if current_reasoning else None
                        )
                        obs = (
                            Observation(results=list(current_observations))
                            if current_observations
                            else None
                        )
                        message_content = (
                            "\n".join(current_text)
                            if current_text
                            else (reasoning_content or "")
                        )

                        steps.append(
                            Step(
                                step_id=step_id,
                                source="agent",
                                message=message_content,
                                reasoning_content=reasoning_content,
                                tool_calls=list(current_tools)
                                if current_tools
                                else None,
                                observation=obs,
                                model_name=self.model_name or "google/gemini-3-flash",
                            )
                        )
                        step_id += 1
                        current_reasoning.clear()
                        current_tools.clear()
                        current_observations.clear()
                        current_text.clear()

                for part in parts:
                    ptype = part.get("type")
                    if ptype == "step-start":
                        _flush_step()
                    elif ptype == "reasoning":
                        current_reasoning.append(part.get("text", ""))
                    elif ptype == "text":
                        current_text.append(part.get("text", ""))
                    elif ptype and ptype.startswith("tool-"):
                        tool_name = ptype[5:]  # remove "tool-"
                        tool_call_id = part.get("toolCallId", "")
                        args = part.get("input", {})
                        current_tools.append(
                            ToolCall(
                                tool_call_id=tool_call_id,
                                function_name=tool_name,
                                arguments=args,
                            )
                        )
                        output_data = part.get("output")
                        if output_data is not None:
                            if (
                                isinstance(output_data, dict)
                                and "output" in output_data
                            ):
                                obs_content = output_data["output"]
                            else:
                                obs_content = str(output_data)
                            current_observations.append(
                                ObservationResult(
                                    source_call_id=tool_call_id,
                                    content=obs_content,
                                )
                            )

                _flush_step()

        if not steps:
            return None

        final_metrics = FinalMetrics(
            total_prompt_tokens=total_input_tokens,
            total_completion_tokens=total_output_tokens,
            total_cached_tokens=total_cached_tokens,
            total_steps=len(steps),
            extra={
                "total_tokens": total_total_tokens,
                "system_tokens": total_system_tokens,
                "tools_tokens": total_tools_tokens,
            },
        )

        trajectory = Trajectory(
            schema_version="ATIF-v1.6",
            session_id=session_id,
            agent=Agent(
                name="pochi",
                version="unknown",
                model_name=self.model_name or "google/gemini-3.1-pro",
            ),
            steps=steps,
            final_metrics=final_metrics,
        )
        return trajectory

    def populate_context_post_run(self, context: AgentContext) -> None:
        pochi_trajectory = self.logs_dir / "pochi" / "trajectory.jsonl"
        if not pochi_trajectory.exists():
            print(f"Pochi trajectory not found at {pochi_trajectory}")
            return

        try:
            log_lines = pochi_trajectory.read_text().splitlines()
        except Exception as e:
            print(f"Error loading Pochi log: {e}")
            return

        # Calculate token counts for context
        n_input_tokens = 0
        n_output_tokens = 0
        n_cache_tokens = 0

        for line in log_lines:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                if msg.get("role") == "assistant":
                    metadata = msg.get("metadata", {})
                    # Pochi log only has totalTokens in metadata currently
                    n_output_tokens += metadata.get("totalTokens", 0)
            except Exception:
                continue

        context.n_input_tokens = n_input_tokens
        context.n_output_tokens = n_output_tokens
        context.n_cache_tokens = n_cache_tokens

        try:
            atif_trajectory = self._convert_pochi_to_atif(log_lines)
            if atif_trajectory:
                atif_path = self.logs_dir / "trajectory.json"
                with open(atif_path, "w") as f:
                    json.dump(atif_trajectory.to_json_dict(), f, indent=2)
        except Exception as e:
            print(f"Error converting Pochi trajectory to ATIF: {e}")

        print(f"done saving logs to dir {self.logs_dir}")
