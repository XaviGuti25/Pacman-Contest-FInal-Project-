import random
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point, manhattan_distance


DIRS = [(Directions.NORTH, (0, 1)), (Directions.SOUTH, (0, -1)),
        (Directions.EAST, (1, 0)), (Directions.WEST, (-1, 0))]


def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    tracker = {}
    a = eval(first)(first_index)
    b = eval(second)(second_index)
    a.tracker = tracker
    b.tracker = tracker
    return [a, b]


class BaseAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.tracker = {}

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        self.width, self.height = walls.width, walls.height
        self.cells = [(x, y) for x in range(self.width) for y in range(self.height) if not walls[x][y]]
        self.neighbors = {}
        for (x, y) in self.cells:
            self.neighbors[(x, y)] = [((x + dx, y + dy), a) for a, (dx, dy) in DIRS
                                      if not walls[x + dx][y + dy]]
        mid = self.width // 2
        self.mid = mid
        bx = mid - 1 if self.red else mid
        self.border = [(bx, y) for y in range(self.height) if not walls[bx][y]]
        self.dead_depth = self.compute_dead_ends()
        self.last_defended = self.get_food_you_are_defending(game_state).as_list()
        self.recent_eaten = None
        for opp in self.get_opponents(game_state):
            self.tracker[opp] = {game_state.get_initial_agent_position(opp): 1.0}

    def is_home(self, cell):
        return cell[0] < self.mid if self.red else cell[0] >= self.mid

    def compute_dead_ends(self):
        deg = {c: len(self.neighbors[c]) for c in self.cells}
        removed = set()
        q = deque(c for c in self.cells if deg[c] <= 1)
        while q:
            c = q.popleft()
            if c in removed:
                continue
            removed.add(c)
            for n, _ in self.neighbors[c]:
                if n not in removed:
                    deg[n] -= 1
                    if deg[n] <= 1:
                        q.append(n)
        depth = {c: 0 for c in self.cells if c not in removed}
        q = deque(depth.keys())
        while q:
            c = q.popleft()
            for n, _ in self.neighbors[c]:
                if n not in depth:
                    depth[n] = depth[c] + 1
                    q.append(n)
        for c in self.cells:
            depth.setdefault(c, 0)
        return depth

    def pos(self, game_state, index):
        p = game_state.get_agent_position(index)
        return None if p is None else nearest_point(p)

    def dist(self, a, b):
        return self.get_maze_distance(a, b)

    def first_step(self, start, goals, blocked):
        goals = set(goals)
        if not goals or start in goals:
            return None
        prev = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c in goals:
                while prev[c][0] != start:
                    c = prev[c][0]
                return prev[c][1]
            for n, a in self.neighbors[c]:
                if n not in prev and (n not in blocked or n in goals):
                    prev[n] = (c, a)
                    q.append(n)
        return None

    def update_beliefs(self, gs):
        my_pos = self.pos(gs, self.index)
        team = [self.pos(gs, i) for i in self.get_team(gs)]
        team = [t for t in team if t is not None]
        noisy = gs.get_agent_distances()
        mover = (self.index - 1) % gs.get_num_agents()
        for opp in self.get_opponents(gs):
            p = self.pos(gs, opp)
            if p is not None:
                self.tracker[opp] = {p: 1.0}
                continue
            b = self.tracker.get(opp, {})
            if opp == mover:
                nb = {}
                for c, w in b.items():
                    opts = [c] + [n for n, _ in self.neighbors[c]]
                    share = w / len(opts)
                    for o in opts:
                        nb[o] = nb.get(o, 0) + share
                b = nb
            is_pac = gs.get_agent_state(opp).is_pacman
            nb = {}
            for c, w in b.items():
                if self.is_home(c) != is_pac:
                    continue
                if any(manhattan_distance(c, t) <= 5 for t in team):
                    continue
                if noisy is not None:
                    w *= gs.get_distance_prob(manhattan_distance(c, my_pos), noisy[opp])
                if w > 0:
                    nb[c] = w
            total = sum(nb.values())
            if total == 0:
                nb = {gs.get_initial_agent_position(opp): 1.0}
            else:
                nb = {c: w / total for c, w in nb.items()}
            self.tracker[opp] = nb

    def likely_pos(self, opp):
        b = self.tracker.get(opp)
        if not b:
            return None
        return max(b.items(), key=lambda kv: kv[1])[0]

    def safe_action(self, gs, action, fallback):
        legal = gs.get_legal_actions(self.index)
        if action in legal:
            return action
        return fallback(gs, legal)


class OffensiveAgent(BaseAgent):

    CARRY_LIMIT = 6

    def threats(self, gs):
        res = []
        for opp in self.get_opponents(gs):
            st = gs.get_agent_state(opp)
            p = self.pos(gs, opp)
            if p is None or st.is_pacman or st.scared_timer > 3:
                continue
            res.append(p)
        return res

    def safe_food(self, food, my_pos, threats):
        if not threats:
            return food
        res = []
        for f in food:
            md = self.dist(my_pos, f)
            d = self.dead_depth[f]
            ok = True
            for t in threats:
                gd = self.dist(t, f)
                if (d == 0 and gd <= md) or (d > 0 and gd - d <= md + d + 1):
                    ok = False
                    break
            if ok:
                res.append(f)
        return res

    def evade(self, gs, legal):
        my_pos = self.pos(gs, self.index)
        threats = self.threats(gs)
        best, best_v = None, None
        for a in legal:
            succ = gs.generate_successor(self.index, a)
            p = nearest_point(succ.get_agent_position(self.index))
            td = min([self.dist(p, t) for t in threats], default=99)
            hd = min(self.dist(p, b) for b in self.border)
            v = (min(td, 6), -self.dead_depth[p], -hd, a != Directions.STOP)
            if best_v is None or v > best_v:
                best, best_v = a, v
        return best if best is not None else random.choice(legal)

    def choose_action(self, gs):
        self.update_beliefs(gs)
        my_pos = self.pos(gs, self.index)
        st = gs.get_agent_state(self.index)
        carrying = st.num_carrying
        food = self.get_food(gs).as_list()
        caps = self.get_capsules(gs)
        threats = self.threats(gs)
        blocked = set()
        for t in threats:
            blocked.add(t)
            blocked.update(n for n, _ in self.neighbors[t])
        home_d = min(self.dist(my_pos, b) for b in self.border)
        moves_left = gs.data.timeleft // 4
        threat_d = min([self.dist(my_pos, t) for t in threats], default=99)

        if not st.is_pacman and st.scared_timer == 0:
            near_inv = [self.pos(gs, o) for o in self.get_opponents(gs)
                        if gs.get_agent_state(o).is_pacman and self.pos(gs, o) is not None]
            near_inv = [p for p in near_inv if self.dist(my_pos, p) <= 2]
            if near_inv:
                a = self.first_step(my_pos, near_inv, set(c for c in self.cells if not self.is_home(c)))
                if a is not None:
                    return self.safe_action(gs, a, self.evade)

        goals = None
        if carrying > 0 and (len(food) <= 2 or moves_left <= home_d + 6):
            goals = self.border
        elif threats and threat_d <= 5 and st.is_pacman:
            safe_caps = [c for c in caps if all(self.dist(my_pos, c) < self.dist(t, c) for t in threats)]
            if safe_caps:
                goals = safe_caps
            elif carrying > 0:
                goals = self.border
            else:
                goals = self.safe_food(food, my_pos, threats) or self.border
        if goals is None and carrying >= self.CARRY_LIMIT:
            near = min([self.dist(my_pos, f) for f in food], default=99)
            if not (near == 1 and not threats):
                goals = self.border
        if goals is None:
            goals = self.safe_food(food, my_pos, threats) or self.border

        a = self.first_step(my_pos, goals, blocked)
        if a is None:
            return self.evade(gs, gs.get_legal_actions(self.index))
        return self.safe_action(gs, a, self.evade)


class DefensiveAgent(BaseAgent):

    def choose_action(self, gs):
        self.update_beliefs(gs)
        my_pos = self.pos(gs, self.index)
        st = gs.get_agent_state(self.index)
        defended = self.get_food_you_are_defending(gs).as_list()
        eaten = set(self.last_defended) - set(defended)
        self.last_defended = defended
        if eaten:
            self.recent_eaten = min(eaten, key=lambda f: self.dist(my_pos, f))
        if self.recent_eaten == my_pos:
            self.recent_eaten = None

        opps = self.get_opponents(gs)
        invaders = [self.pos(gs, o) for o in opps
                    if gs.get_agent_state(o).is_pacman and self.pos(gs, o) is not None]
        blocked = set(c for c in self.cells if not self.is_home(c))
        scared = st.scared_timer > 0

        if invaders:
            self.recent_eaten = None
            if scared:
                for inv in invaders:
                    blocked.add(inv)
                    blocked.update(n for n, _ in self.neighbors[inv])
                target = min(invaders, key=lambda p: self.dist(my_pos, p))
                goals = [c for c in self.cells if self.is_home(c) and self.dist(c, target) == 2]
            else:
                goals = invaders
        elif self.recent_eaten is not None:
            goals = [self.recent_eaten]
        else:
            pac_opps = [o for o in opps if gs.get_agent_state(o).is_pacman]
            if pac_opps:
                goals = [p for p in (self.likely_pos(o) for o in pac_opps) if p is not None]
            else:
                guesses = [p for p in (self.likely_pos(o) for o in opps) if p is not None]
                if guesses:
                    e = min(guesses, key=lambda g: min(self.dist(g, b) for b in self.border))
                    goals = [min(self.border, key=lambda b: self.dist(b, e))]
                else:
                    goals = [min(self.border, key=lambda b: sum(self.dist(b, f) for f in defended))]

        a = self.first_step(my_pos, goals, blocked)
        if a is None:
            return self.hold(gs, gs.get_legal_actions(self.index))
        return self.safe_action(gs, a, self.hold)

    def hold(self, gs, legal):
        my_pos = self.pos(gs, self.index)
        st = gs.get_agent_state(self.index)
        invaders = [self.pos(gs, o) for o in self.get_opponents(gs)
                    if gs.get_agent_state(o).is_pacman and self.pos(gs, o) is not None]
        options = []
        for a in legal:
            succ = gs.generate_successor(self.index, a)
            p = nearest_point(succ.get_agent_position(self.index))
            if not self.is_home(p):
                continue
            if st.scared_timer > 0 and invaders:
                v = min(self.dist(p, i) for i in invaders)
                options.append((v >= 2, -abs(v - 2), a))
            else:
                options.append((True, 0, a))
        if not options:
            return Directions.STOP if Directions.STOP in legal else random.choice(legal)
        best = max(o[:2] for o in options)
        return random.choice([o[2] for o in options if o[:2] == best])
