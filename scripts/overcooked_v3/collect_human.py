"""Play the real Overcooked V3 environment and record human BC demonstrations."""

import argparse
import inspect
import os
import time
from collections import deque
from pathlib import Path

# A desktop collector should not reserve accelerator memory or start an MPL UI.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

import jaxmarl.environments.overcooked_v3 as overcooked_v3  # noqa: E402
from jaxmarl import make  # noqa: E402
from jaxmarl.environments.overcooked_v3.common import (  # noqa: E402
    OvercookedActionsEnum as Action,
)
from jaxmarl.environments.overcooked_v3.human_data import (  # noqa: E402
    ACTION_NAMES,
    HumanEpisode,
)

SUPPORTS_PHASE_ORDER = (
    "phase_order_split" in inspect.signature(overcooked_v3.OvercookedV3).parameters
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        default="split_0",
        choices=tuple(overcooked_v3.ROLE_SCENARIO_LAYOUT_NAMES)
        + tuple(getattr(overcooked_v3, "HARD_SCENARIO_LAYOUT_NAMES", ())),
    )
    parser.add_argument("--mode", choices=("step", "realtime"), default="step")
    parser.add_argument(
        "--hz",
        type=float,
        default=5,
        help="Real-time environment steps per second (1-15)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Randomize initial positions and facing within each agent's connected room",
    )
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument(
        "--phase-order-split", choices=("train", "eval"), default="train"
    )
    parser.add_argument(
        "--players",
        nargs=2,
        default=["anonymous-red", "anonymous-blue"],
        metavar=("RED", "BLUE"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/human"))
    args = parser.parse_args()
    if not 1 <= args.hz <= 15:
        parser.error("--hz must be between 1 and 15")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    if args.phase_order_split == "eval" and not SUPPORTS_PHASE_ORDER:
        parser.error(
            "This environment version does not support Hard phase-order splits"
        )
    return args


class CollectorApp:
    def __init__(self, args, pygame):
        from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

        self.pg = pygame
        self.args = args
        self.env_kwargs = {
            "layout": args.layout,
            "max_steps": args.max_steps,
            "random_agent_positions": args.random_start,
            "observation_type": "default",
            "include_transition_countdown": True,
            "include_layout_change_mask": True,
            "transition_observer": "both",
        }
        if SUPPORTS_PHASE_ORDER:
            self.env_kwargs["phase_order_split"] = args.phase_order_split
        self.env = make("overcooked_v3", **self.env_kwargs)
        self.visualizer = OvercookedV3Visualizer(
            tile_size=32,
            subdivs=1,
            seconds_per_step=1 / args.hz,
            transition_warning_steps=self.env.transition_warning_steps,
        )
        self.pg.init()
        self.screen = self.pg.display.set_mode((1140, 780))
        self.pg.display.set_caption("Overcooked V3 | Human demonstrations")
        self.font = self.pg.font.Font(None, 25)
        self.small = self.pg.font.Font(None, 21)
        self.title = self.pg.font.Font(None, 38)
        self.clock = self.pg.time.Clock()
        self.bindings = {
            self.pg.K_w: (0, Action.up),
            self.pg.K_s: (0, Action.down),
            self.pg.K_a: (0, Action.left),
            self.pg.K_d: (0, Action.right),
            self.pg.K_q: (0, Action.interact),
            self.pg.K_e: (0, Action.stay),
            self.pg.K_UP: (1, Action.up),
            self.pg.K_DOWN: (1, Action.down),
            self.pg.K_LEFT: (1, Action.left),
            self.pg.K_RIGHT: (1, Action.right),
            self.pg.K_RSHIFT: (1, Action.interact),
            self.pg.K_SLASH: (1, Action.stay),
        }
        self.buttons = {}
        self.episode_number = 0
        self.episode = None
        self.message = ""
        self.running = True
        self.new_episode()

    def clear_input(self):
        self.pending = [None, None]
        self.input_queues = [deque(), deque()]
        self.held = []

    def press_action(self, key):
        # OS key repeats must not enqueue multiple interactions for one press.
        if key in self.held:
            return
        self.held.append(key)
        agent, action = self.bindings[key]
        if self.args.mode == "realtime":
            # Preserve both facing and interaction when tapped within one tick.
            # KEYUP releases held movement but never cancels a queued tap.
            self.input_queues[agent].append(action)
        else:
            self.pending[agent] = action

    def new_episode(self):
        if self.episode is not None and not self.episode.closed:
            self.episode.finish("pending")
        self.paused = True
        self.clear_input()
        self.message = "Preparing kitchen... first render can take a moment."
        self.screen.fill((20, 25, 34))
        self.text(self.message, 35, 35)
        self.pg.display.flip()
        seed = (self.args.seed + self.episode_number) % 2**32
        self.episode = HumanEpisode(
            self.env,
            self.env_kwargs,
            self.args.output,
            seed,
            self.args.mode,
            self.args.hz,
            self.args.players,
        )
        self.episode.metadata["input_policy"] = (
            "fifo_keydown_v1"
            if self.args.mode == "realtime"
            else "last_selected_action"
        )
        self.episode_number += 1
        self.refresh_frame()
        # Clear input accumulated during JAX compilation, before play begins.
        self.pg.event.clear([self.pg.KEYDOWN, self.pg.KEYUP])
        self.started_at = time.monotonic()
        self.next_tick = self.started_at
        self.message = (
            "Queue actions, then SPACE."
            if self.args.mode == "step"
            else "SPACE to start. Both players are recorded."
        )
        print(
            f"Ready: {self.args.layout} / {self.args.mode}. {self.message}", flush=True
        )
        print(
            f"Start: {'random' if self.args.random_start else 'fixed'} / "
            f"seed={seed} / positions="
            f"{np.column_stack((self.episode.state.agents.pos.x, self.episode.state.agents.pos.y)).tolist()}",
            flush=True,
        )

    def refresh_frame(self):
        frame = np.ascontiguousarray(
            self.visualizer.render_sequence([self.episode.state])[0]
        )
        surface = self.pg.image.frombuffer(
            frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB"
        )
        scale = min(680 / frame.shape[1], 460 / frame.shape[0])
        size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
        self.frame = self.pg.transform.scale(surface, size)

    def advance(self):
        if self.episode.closed:
            return
        actions = [Action.stay, Action.stay]
        labels = [False, False]
        for agent in range(2):
            if self.args.mode == "realtime":
                if self.input_queues[agent]:
                    actions[agent] = self.input_queues[agent][0]
                else:
                    for key in self.held:
                        owner, action = self.bindings[key]
                        if owner == agent and action in (
                            Action.up,
                            Action.down,
                            Action.left,
                            Action.right,
                        ):
                            actions[agent] = action
                # While running, both humans control their agent; idle is a choice.
                labels[agent] = True
            elif self.pending[agent] is not None:
                actions[agent] = self.pending[agent]
                labels[agent] = True
        self.episode.step(actions, labels, time.monotonic() - self.started_at)
        for queue in self.input_queues:
            if queue:
                queue.popleft()
        self.pending = [None, None]
        self.refresh_frame()
        if self.episode.closed:
            self.paused = True
            self.clear_input()
            self.message = "Episode saved. K: keep for BC / X: reject / N: next."

    def command(self, command):
        if command == "advance":
            if self.episode.closed:
                return
            if self.args.mode == "step":
                self.advance()
            else:
                self.paused = not self.paused
                self.clear_input()
                self.next_tick = time.monotonic() + 1 / self.args.hz
                self.message = (
                    "Paused (no samples recorded)."
                    if self.paused
                    else "Recording both players. SPACE to pause."
                )
        elif command == "save":
            self.episode.save()
            self.message = "Checkpoint saved. Recording can continue."
        elif command in ("keep", "reject"):
            if not self.episode.length:
                self.message = "Play at least one step before saving a demonstration."
                return
            status = "accepted" if command == "keep" else "rejected"
            path = self.episode.finish(status)
            self.paused = True
            self.clear_input()
            self.message = f"{status.upper()} saved. N starts the next episode."
            print(f"{status}: {path}", flush=True)
        elif command == "next":
            self.new_episode()

    def text(self, message, x, y, color=(222, 230, 241), font=None):
        self.screen.blit((font or self.font).render(message, True, color), (x, y))

    def draw(self):
        self.screen.fill((20, 25, 34))
        self.text("OVERCOOKED / HUMAN LAB", 28, 25, font=self.title)
        self.text(
            f"{self.args.layout}  /  {self.args.mode.upper()}  /  episode {self.episode_number}"
            f"  /  {'RANDOM' if self.args.random_start else 'FIXED'} START"
            f"  /  seed {self.episode.metadata['seed']}",
            30,
            69,
            color=(139, 157, 182),
        )
        state = self.episode.state
        self.text(
            f"STEP {self.episode.length:03d} / {self.args.max_steps}       SCORE {self.episode.score:g}       PHASE {int(state.layout_index)}",
            30,
            111,
        )
        self.screen.blit(
            self.frame,
            (
                30 + (680 - self.frame.get_width()) // 2,
                166 + (460 - self.frame.get_height()) // 2,
            ),
        )
        self.text(
            self.visualizer.caption_with_countdown(state),
            30,
            643,
            color=(245, 189, 87),
            font=self.small,
        )
        self.pg.draw.rect(
            self.screen, (29, 36, 48), (745, 100, 366, 552), border_radius=14
        )
        self.text("CONTROLS", 768, 121)
        self.text("RED / agent_0", 768, 167, color=(255, 122, 122))
        self.text("W A S D   move", 768, 199)
        self.text("Q   interact       E   wait", 768, 229)
        self.text("BLUE / agent_1", 768, 276, color=(126, 176, 255))
        self.text("Arrow keys   move", 768, 308)
        self.text("Right Shift   interact       /   wait", 768, 338, font=self.small)
        for index, y in enumerate((385, 416)):
            queue = self.input_queues[index]
            action = queue[0] if queue else self.pending[index]
            label = ACTION_NAMES[int(action)].upper() if action is not None else "-"
            remaining = f" (+{len(queue) - 1})" if len(queue) > 1 else ""
            self.text(f"Queued {index}: {label}{remaining}", 768, y)
        self.text("SPACE  step / start / pause", 768, 463, font=self.small)
        self.text("F5  checkpoint    N  next episode", 768, 490, font=self.small)
        self.text("K  finish + keep    X  finish + reject", 768, 517, font=self.small)
        self.text("ESC  save pending + quit", 768, 544, font=self.small)
        self.text("Face a station, then interact.", 768, 590, font=self.small)
        self.text("Fill pot > wait > plate > serve.", 768, 617, font=self.small)
        status = self.episode.metadata["status"].upper()
        self.text(f"{status}  |  {self.message}", 30, 686, font=self.small)
        labels = [
            (
                "advance",
                "SPACE / step"
                if self.args.mode == "step"
                else ("SPACE / start" if self.paused else "SPACE / pause"),
            ),
            ("keep", "K / keep for BC"),
            ("reject", "X / reject"),
            ("next", "N / next episode"),
        ]
        for i, (command, label) in enumerate(labels):
            rect = self.pg.Rect(30 + i * 274, 725, 258, 35)
            self.buttons[command] = rect
            self.pg.draw.rect(self.screen, (45, 61, 80), rect, border_radius=7)
            self.text(label, rect.x + 14, rect.y + 8, font=self.small)
        self.pg.display.flip()

    def run(self):
        try:
            while self.running:
                for event in self.pg.event.get():
                    if event.type == self.pg.QUIT:
                        self.running = False
                    elif event.type == self.pg.WINDOWFOCUSLOST:
                        self.paused = True
                        self.clear_input()
                        self.message = (
                            "Window unfocused. Input cleared; real-time paused."
                        )
                    elif event.type == self.pg.KEYUP:
                        if event.key in self.held:
                            self.held.remove(event.key)
                    elif event.type == self.pg.KEYDOWN:
                        if getattr(event, "repeat", False):
                            continue
                        if event.key == self.pg.K_ESCAPE:
                            self.running = False
                        elif event.key in self.bindings and not self.episode.closed:
                            if self.args.mode == "step" or not self.paused:
                                self.press_action(event.key)
                        else:
                            command = {
                                self.pg.K_SPACE: "advance",
                                self.pg.K_F5: "save",
                                self.pg.K_k: "keep",
                                self.pg.K_x: "reject",
                                self.pg.K_n: "next",
                            }.get(event.key)
                            if command:
                                self.command(command)
                    elif event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                        for command, rect in self.buttons.items():
                            if rect.collidepoint(event.pos):
                                self.command(command)
                                break
                if (
                    self.running
                    and self.args.mode == "realtime"
                    and not self.paused
                    and not self.episode.closed
                    and time.monotonic() >= self.next_tick
                ):
                    self.advance()
                    # Never catch up with a burst of unobserved actions after lag.
                    self.next_tick = time.monotonic() + 1 / self.args.hz
                self.draw()
                self.clock.tick(60)
        finally:
            try:
                if not self.episode.closed:
                    self.episode.finish("pending")
                else:
                    self.episode.save()
                if self.episode.length:
                    print(f"Saved: {self.episode.path}", flush=True)
            finally:
                self.pg.quit()


def main():
    args = parse_args()
    try:
        import pygame
    except ImportError as error:
        raise SystemExit(
            'Install the collector first: python -m pip install -e ".[human]"'
        ) from error
    print(
        "Opening Overcooked V3. Initial JAX compilation may take a moment.", flush=True
    )
    try:
        CollectorApp(args, pygame).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
