from collections import deque
from itertools import combinations
from helper_types import Action, Direction, Percept, Room, PossibleWorld
from utils import is_facing_wampa, orientation_to_delta
from typing import List, Dict, Set, Optional


# KNOWLEDGE BASE
class KB:
    def __init__(self, agent):
        # set of rooms that are known to exist
        self.all_rooms: Set[Room] = {agent.loc}
        # set of rooms that are known to be safe
        self.safe_rooms: Set[Room] = {agent.loc}
        # set of visited rooms (x, y)
        self.visited_rooms: Set[Room] = {agent.loc}
        #
        # set of rooms where stench has been perceived
        self.stench: Set[Room] = set()
        # set of rooms where breeze has been perceived
        self.breeze: Set[Room] = set()
        # {loc: direction} where bump has been perceived
        self.bump: Dict[Room, str] = dict()
        self.gasp: bool = False  # True if gasp has been perceived
        self.scream: bool = False  # True if scream has been perceived
        #
        # set of rooms (x, y) that are known to be walls
        self.walls: Set[Room] = set()
        # set of rooms (x, y) that are known to be pits
        self.pits: Set[Room] = set()
        # room (x, y) that is known to be the Wampa
        self.wampa: Optional[Room] = None
        # room (x, y) that is known to be Luke
        self.luke: Optional[Room] = None


# AGENT
class Agent:
    def __init__(self, world):
        self.world = world
        self.loc = (0, 0)
        self.direction = Direction.UP
        self.blaster = True
        self.has_luke = False
        self.KB = KB(self)

    @staticmethod
    def adjacent_rooms(room):
        """Returns a set of tuples representing all possible adjacent rooms to
        'room' Use this function to update KB.all_rooms."""
        x, y = room
        return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}

    def record_percepts(self, sensed_percepts, current_location):
        """Update the percepts in agent's KB with the percepts sensed in the
        current location, and update visited_rooms and all_rooms."""
        self.loc = current_location
        present_percepts = set(p for p in sensed_percepts if p)
        kb = self.KB
        kb.visited_rooms.add(self.loc)
        kb.safe_rooms.add(self.loc)
        kb.all_rooms.update(self.adjacent_rooms(self.loc) | {self.loc})
        if Percept.STENCH in present_percepts and not kb.scream:
            kb.stench.add(self.loc)
        if Percept.BREEZE in present_percepts:
            kb.breeze.add(self.loc)
        if Percept.BUMP in present_percepts:
            kb.bump[self.loc] = self.direction
        kb.gasp = Percept.GASP in present_percepts
        kb.scream = kb.scream or Percept.SCREAM in present_percepts

    @staticmethod
    def enumerate_possible_worlds(
        all_rooms: Set[Room], safe_rooms: Set[Room], walls: Set[Room],
    ) -> Set[PossibleWorld]:
        """Return the set of all possible worlds, where a possible world is a
        tuple of (pit_rooms, wampa_room), pit_rooms is a tuple of tuples
        representing possible pit rooms, and wampa_room is a tuple
        representing a possible wampa room.

        Since the goal is to combinatorially enumerate all the possible worlds
        (pit and wampa locations) over the set of rooms that could potentially
        have a pit or a wampa, we first want to find that set.
        To do that, subtract the set of rooms that you know cannot
        have a pit or wampa from the set of all rooms.
        For example, a room with a wall cannot have a pit or wampa.

        Then use itertools.combinations to return the set of possible worlds,
        or all combinations of possible pit and wampa locations.

        You may find the utils.flatten(tup) method useful here for flattening
        wampa_room from a tuple of tuples into a tuple.

        The output of this function will be queried to find the model of the
        query, and will be checked for consistency with the KB
        to find the model of the KB."""
        candidates = sorted(all_rooms - safe_rooms - walls)
        worlds = set()
        for count in range(len(candidates) + 1):
            for rooms in combinations(candidates, count):
                pits = frozenset(rooms)
                worlds.add((pits, None))
                for room in candidates:
                    if room not in pits:
                        worlds.add((pits, room))
        return worlds

    def pit_room_is_consistent_with_KB(self, pit_room: Optional[Room]) -> bool:
        """Return True if the room could be a pit given breeze in KB, False
        otherwise. A room could be a pit if all adjacent rooms that have been
        visited have had breeze perceived in them.
        A room cannot be a pit if any adjacent rooms that have been visited
        have not had breeze perceived in them.
        This will be used to find the model of the KB."""
        if pit_room is None:  # It is possible that there are no pits
            return not self.KB.breeze  # if no breeze has been perceived yet
        return self.adjacent_rooms(
            pit_room) & self.KB.visited_rooms <= self.KB.breeze

    def wampa_room_is_consistent_with_KB(
            self, wampa_room: Optional[Room]) -> bool:
        """Return True if the room could be a wampa given stench in KB, False
        otherwise. A room could be a wampa if all adjacent rooms that have
        been visited have had stench perceived in them.
        A room cannot be a wampa if any adjacent rooms that have been
        visited have not had stench perceived in them.
        This will be used to find the model of the KB."""
        if wampa_room is None:  # It is possible that there is no Wampa
            return not self.KB.stench  # if no stench has been perceived yet
        adjacent = self.adjacent_rooms(wampa_room)
        return (not self.KB.scream
                and adjacent & self.KB.visited_rooms == self.KB.stench)

    def find_model_of_KB(
            self,
            possible_worlds: Set[PossibleWorld]) -> Set[PossibleWorld]:
        """Return the subset of all possible worlds consistent with KB.
        possible_worlds is a set of tuples (pit_rooms, wampa_room),
        pit_rooms is a set of tuples of possible pit rooms,
        and wampa_room is a tuple representing a possible wampa room.
        A world is consistent with the KB if wampa_room is consistent
        and all pit rooms are consistent with the KB."""
        models = set()
        for pits, wampa in possible_worlds:
            if not self.wampa_room_is_consistent_with_KB(wampa):
                continue
            if not all(self.pit_room_is_consistent_with_KB(pit)
                       for pit in pits):
                continue
            # Every observed breeze must have at least one adjacent pit.
            if all(self.adjacent_rooms(room) & set(pits)
                   for room in self.KB.breeze):
                models.add((pits, wampa))
        return models

    @staticmethod
    def find_model_of_query(
        query: str, room: Room, possible_worlds: Set[PossibleWorld]
    ) -> Set[PossibleWorld]:
        """Where query can be "pit_in_room", "wampa_in_room", "no_pit_in_room"
        or "no_wampa_in_room",filter the set of worlds
        according to the query and room."""
        predicates = {
            "pit_in_room": lambda pits, wampa: room in pits,
            "wampa_in_room": lambda pits, wampa: room == wampa,
            "no_pit_in_room": lambda pits, wampa: room not in pits,
            "no_wampa_in_room": lambda pits, wampa: room != wampa,
        }
        if query not in predicates:
            raise ValueError(f"Unknown query: {query}")
        return {
            world for world in possible_worlds if predicates[query](
                *world)}

    def infer_wall_locations(self):
        """If a bump is perceived, infer wall locations along the entire known
        length of the room."""
        min_x = min(self.KB.all_rooms, key=lambda x: x[0])[0]
        max_x = max(self.KB.all_rooms, key=lambda x: x[0])[0]
        min_y = min(self.KB.all_rooms, key=lambda x: x[1])[1]
        max_y = max(self.KB.all_rooms, key=lambda x: x[1])[1]
        for room, orientation in self.KB.bump.items():
            if orientation == Direction.UP:
                for x in range(min_x, max_x + 1, 1):
                    self.KB.walls.add((x, room[1] + 1))
            elif orientation == Direction.DOWN:
                for x in range(min_x, max_x + 1, 1):
                    self.KB.walls.add((x, room[1] - 1))
            elif orientation == Direction.LEFT:
                for y in range(min_y, max_y + 1, 1):
                    self.KB.walls.add((room[0] - 1, y))
            elif orientation == Direction.RIGHT:
                for y in range(min_y, max_y + 1, 1):
                    self.KB.walls.add((room[0] + 1, y))

    def basic_forward_chaining(self):
        """First, make some simple inferences using forward chaining:
        1. If there is no breeze or stench in current location, infer that the
        adjacent rooms are safe.
        2. Infer wall locations given bump percept.
        3. Infer Luke's location given gasp percept.
        4. Infer whether the Wampa is alive given scream percept. Clear stench
        from the KB if Wampa is dead.
        """
        kb = self.KB
        if kb.scream:
            # A Wampa room cannot also contain a pit.
            if kb.wampa is not None:
                kb.safe_rooms.add(kb.wampa)
            kb.stench.clear()
            kb.wampa = None
        if self.loc not in kb.breeze and self.loc not in kb.stench:
            kb.safe_rooms.update(self.adjacent_rooms(self.loc))
        self.infer_wall_locations()
        kb.safe_rooms.difference_update(kb.walls)
        if kb.gasp:
            kb.luke = self.loc

    def backward_chaining_resolution(self):
        """
        Infer whether each adjacent room is safe, pit or wampa by
        following the backward-chaining resolution algorithm:
        1. Enumerate possible worlds.
        2. Find the model of the KB, i.e. the subset of possible worlds
        consistent with the KB.
        3. For each adjacent room and each query, find the model of the query.
        4. If the model of the KB is a subset of the model of the query, the
        query is entailed by the KB.
        5. Update KB.pits, KB.wampa, and KB.safe_rooms based on any newly
        derived knowledge.
        """
        kb = self.KB
        candidates = kb.all_rooms - kb.safe_rooms - kb.walls
        # Eliminate impossible individual hazards before the exponential step.
        # This produces the same KB models as filtering the full enumeration.
        pit_candidates = sorted(room for room in candidates
                                if self.pit_room_is_consistent_with_KB(room))
        wampas = [room for room in sorted(candidates)
                  if self.wampa_room_is_consistent_with_KB(room)]
        if self.wampa_room_is_consistent_with_KB(None):
            wampas.append(None)
        worlds = set()
        for count in range(len(pit_candidates) + 1):
            for rooms in combinations(pit_candidates, count):
                pits = frozenset(rooms)
                if not all(self.adjacent_rooms(room) &
                           pits for room in kb.breeze):
                    continue
                for wampa in wampas:
                    if wampa not in pits:
                        worlds.add((pits, wampa))
        models = self.find_model_of_KB(worlds)
        if not models:
            return  # Inconsistent evidence must not entail every query.
        for room in kb.all_rooms - kb.walls:
            entailed = {
                query: models <= self.find_model_of_query(
                    query,
                    room,
                    worlds) for query in (
                    "pit_in_room",
                    "wampa_in_room",
                    "no_pit_in_room",
                    "no_wampa_in_room")}
            if entailed["pit_in_room"]:
                kb.pits.add(room)
            if entailed["wampa_in_room"]:
                kb.wampa = room
            if entailed["no_pit_in_room"] and entailed["no_wampa_in_room"]:
                kb.safe_rooms.add(room)

    def inference_algorithm(self):
        """First, make some basic inferences
        Then, infer whether each adjacent room is safe, pit or wampa by
        following the backward-chaining resolution algorithm
        """
        self.basic_forward_chaining()
        self.backward_chaining_resolution()

    def all_safe_next_actions(self) -> List[Action]:
        """Define R2D2's valid and safe next actions based on his current
        location and knowledge of the environment."""
        safe_actions = [Action.GRAB]
        x, y = self.loc
        dx, dy = orientation_to_delta[self.direction]
        forward_room = (x+dx, y+dy)
        safe_actions = [Action.LEFT, Action.RIGHT]
        if forward_room in self.KB.safe_rooms - self.KB.walls:
            safe_actions.append(Action.FORWARD)
        if self.KB.luke == self.loc and not self.has_luke:
            safe_actions.append(Action.GRAB)
        if self.loc == (0, 0) and self.has_luke:
            safe_actions.append(Action.CLIMB)
        if (self.blaster and not self.KB.scream
                and is_facing_wampa(self.loc, self.direction, self.KB.wampa)):
            safe_actions.append(Action.SHOOT)
        return safe_actions

    def choose_next_action(self) -> Action:
        """Choose next action from all safe next actions. You may want to
        prioritize some actions based on current state. For example, if R2D2
        knows Luke's location and is in the same room as Luke, you may want
        to prioritize 'grab' over all other actions. Similarly, if R2D2 has
        Luke, you may want to prioritize moving toward the exit. You can
        implement this as basically (randomly choosing between safe actions)
        or as sophisticated (optimizing exploration of unvisited states,
        finding shortest paths, etc.) as you like."""
        actions = self.all_safe_next_actions()
        if Action.GRAB in actions:
            return Action.GRAB
        if Action.CLIMB in actions:
            return Action.CLIMB
        if Action.SHOOT in actions:
            return Action.SHOOT

        safe = self.KB.safe_rooms - self.KB.walls
        targets = ({(0, 0)} if self.has_luke else
                   safe - self.KB.visited_rooms)
        if not self.has_luke and self.KB.luke is not None:
            targets = {self.KB.luke}
        directions = [Direction.UP, Direction.RIGHT,
                      Direction.DOWN, Direction.LEFT]
        start = (self.loc, self.direction)
        queue = deque([(start, None)])
        seen = {start}
        while queue:
            (room, direction), first = queue.popleft()
            if first is not None:
                if room in targets:
                    return first
                if (not self.has_luke and self.blaster and not self.KB.scream
                        and is_facing_wampa(room, direction, self.KB.wampa)):
                    return first
            index = directions.index(direction)
            dx, dy = orientation_to_delta[direction]
            forward = (room[0] + dx, room[1] + dy)
            successors = []
            if forward in safe:
                successors.append(((forward, direction), Action.FORWARD))
            successors.extend([
                ((room, directions[(index - 1) % 4]), Action.LEFT),
                ((room, directions[(index + 1) % 4]), Action.RIGHT),
            ])
            for state, action in successors:
                if state not in seen:
                    seen.add(state)
                    queue.append(
                        (state, first if first is not None else action))
        return Action.LEFT


# Approximately how many hours did you spend on this assignment?
feedback_question_1 = """2 Hours"""

# Which aspects of this assignment did you find most challenging?
# Were there any significant stumbling blocks?
feedback_question_2 = (
    "Checking consistency between possible worlds and the KB."
)

# Which aspects of this assignment did you like?
# Is there anything you would have changed?
feedback_question_3 = """All is fine."""
