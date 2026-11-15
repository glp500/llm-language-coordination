"""Official scenario adapter and evaluator metrics; HM still owns agent decisions."""

from pathlib import Path

from controller import record

OUTPUT = None
FRAME_LIMIT = None


class Scenario:
    """Replace the upstream Ray/Gym conversion layer with the same dm_env tuple conversion."""

    def __init__(self, native, factory, output, frame_limit, scenario_name):
        if factory.num_focal_players() != 1:
            raise ValueError("This experiment requires exactly one focal player")
        self.native, self.output, self.frame_limit = native, Path(output), frame_limit
        self.frame, self.returns, self.events = 0, [], {}
        self.native_terminated = False
        self.scenario_name, self.pd_choices = scenario_name, []
        observable = native.observables().substrate
        self.subscriptions = [
            observable.action.subscribe(self.action),
            observable.timestep.subscribe(self.observe),
            observable.events.subscribe(self.event),
        ]

    def action(self, actions):
        """Full-population native actions are evaluator evidence, never added to actor prompts."""
        record(self.output / "native-actions.jsonl", {"frame": self.frame, "actions": actions})

    def observe(self, timestep):
        """Accumulate every player's native return, including background partners."""
        if timestep.reward is None:
            return
        if not self.returns:
            self.returns = [0.0] * len(timestep.reward)
        self.returns = [total + float(reward) for total, reward in zip(self.returns, timestep.reward, strict=True)]
        record(self.output / "native-rewards.jsonl", {"frame": self.frame, "rewards": timestep.reward})

    def event(self, event):
        """Preserve native event payloads; generic event counts do not invent cooperation measures."""
        name, values = event
        self.events[name] = self.events.get(name, 0) + 1
        record(self.output / "native-events.jsonl", {"frame": self.frame, "name": name, "values": values})
        if name == "interaction" and self.scenario_name.startswith("prisoners_dilemma_in_the_matrix__repeated_"):
            fields = {
                key.decode() if isinstance(key, bytes) else key: value
                for key, value in zip(values[1::2], values[2::2], strict=True)
            }
            row, col = fields["row_inventory"], fields["col_inventory"]
            focal, partner = (row, col) if int(fields["row_player_idx"]) == 1 else (col, row)
            if len(focal) != 2 or len(partner) != 2 or sum(focal) <= 0 or sum(partner) <= 0:
                raise ValueError("PD interaction did not contain the native two-resource inventories")
            choice = {
                "frame": self.frame,
                "focal_cooperation_share": float(focal[0] / sum(focal)),
                "partner_cooperation_share": float(partner[0] / sum(partner)),
                "focal_inventory": focal,
                "partner_inventory": partner,
            }
            self.pd_choices.append(choice)
            record(self.output / "pd-cooperation.jsonl", choice)

    def observations(self, timestep):
        """Keep the fork's focal observation permissions, including HM's WORLD.RGB perception input."""
        observations = {"player_0": timestep.observation[0]}
        if "WORLD.RGB" not in observations["player_0"]:
            raise ValueError("Install the pinned HM Melting Pot fork: its WORLD.RGB permission is required")
        return observations

    def reset(self):
        self.frame = 0
        return self.observations(self.native.reset()), {}

    def step(self, actions):
        if set(actions) != {"player_0"}:
            raise ValueError("Expected one action for focal player_0")
        self.frame += 1
        timestep = self.native.step([actions["player_0"]])
        self.native_terminated = timestep.last()
        rewards = {"player_0": float(timestep.reward[0])}
        done = {"__all__": timestep.last() or self.frame >= self.frame_limit}
        return self.observations(timestep), rewards, done, done, {}

    def get_agent_ids(self):
        return {"player_0"}

    def summary(self):
        return {
            "frames": self.frame,
            "per_player_returns": self.returns,
            "collective_return": sum(self.returns) if self.returns else None,
            "per_capita_return": sum(self.returns) / len(self.returns) if self.returns else None,
            "native_event_counts": self.events,
            "population": len(self.returns),
            "native_terminated": self.native_terminated,
            "pd_interactions": len(self.pd_choices) if self.scenario_name.startswith("prisoners_dilemma") else None,
            "pd_cooperation_share": sum(c["focal_cooperation_share"] for c in self.pd_choices) / len(self.pd_choices)
            if self.pd_choices
            else None,
            "pd_partner_cooperation_share": sum(c["partner_cooperation_share"] for c in self.pd_choices) / len(self.pd_choices)
            if self.pd_choices
            else None,
            "return_scope": "whole_native_population_focal_plus_background",
        }

    def close(self):
        for subscription in self.subscriptions:
            subscription.dispose()
        self.native.close()


def scenario_creator(name):
    """Use the original HM fork's scenario factory, background bots and observation permissions."""
    from meltingpot import scenario

    if OUTPUT is None or FRAME_LIMIT is None:
        raise RuntimeError("Set the episode output and native frame limit before constructing HM")
    factory = scenario.get_factory(name)
    return Scenario(factory.build(), factory, OUTPUT, FRAME_LIMIT, name)
