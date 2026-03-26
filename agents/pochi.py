import os
import json
import shlex
import shutil
import subprocess

from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from pydantic import BaseModel
from pathlib import Path

from harbor.models.trajectories import (
    Trajectory,
    Step,
    ToolCall,
    Observation,
    ObservationResult,
    FinalMetrics,
    Agent,
)


class ExecInput(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: int | None = None

class Pochi(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "pochi"

    # this is not used, maybe later version of harbor would use
    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.pochi/bin:$PATH"; pochi --version'

    # this defined which version of pochi to install
    # we have to specify this so that the version could be shown in result.json
    def version(self) -> str | None:
        return "v0.6.5"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()
    
    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-pochi.sh.j2"

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        model = self.model_name if self.model_name else "google/gemini-3-flash"

        env = {
            "POCHI_LOG": "debug",
            "E2B_API_KEY": os.environ.get("E2B_API_KEY", ""),
            "POCHI_API_KEY": os.environ.get("POCHI_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "DEEPINFRA_API_KEY": os.environ.get("DEEPINFRA_API_KEY", ""),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
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
        "moonshotai/Kimi-K2.5": {
          "name": "kimi-K2.5"
        },
        "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
          "name": "qwen3-coder-480b"
        }
      }
    },
    "anthropic": {
      "kind": "openai",
      "baseURL": "https://api.anthropic.com/v1",
      "apiKey": "ANTHROPIC_API_KEY",
      "models": {
        "claude-opus-4-6": {
          "name": "claude-opus-4-6"
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

        return [
            ExecInput(
                command=(
                    "cat << 'EOF' | sed "
                    "-e \"s/OPENAI_API_KEY/${OPENAI_API_KEY}/g\" "
                    "-e \"s/DEEPINFRA_API_KEY/${DEEPINFRA_API_KEY}/g\" "
                    "-e \"s/ANTHROPIC_API_KEY/${ANTHROPIC_API_KEY}/g\" "
                    "> ~/.pochi/config.jsonc\n"
                    f"{config_json}\n"
                    "EOF"
                ),
                env=env,
            ),            
            ExecInput(
                command=(
                    "pochi "
                    f"--model {model} "
                    "--max-steps 200 "
                    "--max-retries 10 "
                    "--blobs-dir /logs/agent/pochi/blobs "
                    "--stream-json /logs/agent/pochi/trajectory.jsonl "
                    "> >(tee /logs/agent/pochi/stdout.txt) "
                    "2> >(tee /logs/agent/pochi/stderr.txt >&2) "
                    "<<'EOF'\n"
                    f"{instruction}\n"
                    "EOF"
                ),
                env=env,
            ),
        ]

    def _convert_pochi_to_atif(
        self, log_lines: list[str]
    ) -> Trajectory | None:
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
                text_parts = [p.get("text", "") for p in msg.get("parts", []) if p.get("type") == "text"]
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
                    nonlocal step_id, current_reasoning, current_tools, current_observations, current_text
                    if current_tools or current_reasoning or current_observations or current_text:
                        reasoning_content = "\n".join(current_reasoning) if current_reasoning else None
                        obs = Observation(results=list(current_observations)) if current_observations else None
                        message_content = "\n".join(current_text) if current_text else (reasoning_content or "")

                        steps.append(
                            Step(
                                step_id=step_id,
                                source="agent",
                                message=message_content,
                                reasoning_content=reasoning_content,
                                tool_calls=list(current_tools) if current_tools else None,
                                observation=obs,
                                model_name=self.model_name or "google/gemini-3.1-pro"
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
                            if isinstance(output_data, dict) and "output" in output_data:
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
