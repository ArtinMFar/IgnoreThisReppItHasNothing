"""Synthetic long-memory tasks with floods of repetitive, low-information content.

Each sample is a chat in which a few important facts are stated early (natural sentences), followed by thousands
of tokens of repetitive content: NPC movement commands, game ticks, sensor telemetry, heartbeat logs, chat spam,
progress bars, price ticks... At the end the user asks about the early facts. Because the facts end up far outside
the 2048-token attention window, the only way to answer is through the recurrent state, and the repetitive content
must not wash them out. Repetitive spans are flagged FLAG_FILLER (priority-gate supervision) and the final answer is
flagged FLAG_CE (ground-truth supervision).
"""

import random

from .formatting import FLAG_CE, FLAG_FILLER, chat_segments, encode_segments

FIRST = ["Lyra", "Garrick", "Mira", "Tobias", "Selene", "Darius", "Aiko", "Ravi", "Elena", "Kofi", "Ingrid", "Mateo",
         "Yuki", "Farah", "Bjorn", "Leila", "Owen", "Priya", "Sven", "Zara", "Nadia", "Hugo", "Amara", "Dmitri",
         "Chen", "Isla", "Tariq", "Freya", "Jonas", "Esme", "Kenji", "Lucia", "Omar", "Rosa", "Felix", "Hana"]
CITIES = ["Eldermoor", "Port Vessa", "Lisbon", "Kyoto", "Nairobi", "Ravenhollow", "Duskwater", "Toronto", "Oslo",
          "Marrakesh", "Brightvale", "Quito", "Stonebridge", "Adelaide", "Frostgate", "Cairo", "Ashford", "Lima"]
JOBS = ["blacksmith", "nurse", "ship captain", "botanist", "software engineer", "cartographer", "baker", "pilot",
        "librarian", "alchemist", "teacher", "architect", "fisherman", "violinist", "detective", "chef"]
COLORS = ["teal", "crimson", "amber", "violet", "olive", "indigo", "coral", "silver", "saffron", "emerald"]
FOODS = ["peanuts", "shellfish", "strawberries", "gluten", "sesame", "mushrooms", "eggs", "kiwi", "soy"]
ITEMS = ["amulet", "silver key", "map fragment", "lantern", "spellbook", "compass", "music box", "dagger",
         "ruby ring", "letter", "pocket watch", "crystal shard", "journal", "rope", "healing potion"]
PLACES = ["loose floorboard", "old oak tree", "well", "anvil", "third bookshelf", "chimney", "rain barrel",
          "statue of the king", "broken cart", "bell tower"]
LOCATIONS = ["tavern", "chapel", "mill", "harbor", "forge", "market square", "library", "lighthouse", "barn"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = ["silver", "heron", "midnight", "copper", "lantern", "willow", "thunder", "velvet", "ember", "frost",
         "maple", "raven", "harbor", "cinder", "orchid", "granite"]
RELATIONS = ["sister", "brother", "daughter", "son", "grandmother", "best friend", "mentor", "cousin"]
PETS = ["cat", "dog", "parrot", "horse", "ferret", "owl", "tortoise"]


def _code(rng):
    return f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(100, 999)}-{rng.randint(10, 99)}"


def make_facts(rng):
    """Returns (statement, question, answer, key, values) tuples; keys allow updates, values are what must be recalled."""
    n1, n2, n3 = rng.sample(FIRST, 3)
    city, job, color, food = rng.choice(CITIES), rng.choice(JOBS), rng.choice(COLORS), rng.choice(FOODS)
    item, place, loc = rng.choice(ITEMS), rng.choice(PLACES), rng.choice(LOCATIONS)
    day, hour = rng.choice(DAYS), f"{rng.randint(1, 12)}:{rng.choice(['00', '15', '30', '45'])}"
    w1, w2 = rng.sample(WORDS, 2)
    code, rel, pet, pet_name = _code(rng), rng.choice(RELATIONS), rng.choice(PETS), rng.choice(FIRST)
    count = rng.randint(3, 17)
    facts = [
        (f"By the way, my name is {n1} and I'm a {job} from {city}.", "What is my name and where am I from?",
         f"Your name is {n1}, and you're a {job} from {city}.", "name", (n1, city)),
        (f"The code for the vault is {code}, please remember it.", "What is the vault code I gave you?",
         f"The vault code is {code}.", "code", (code,)),
        (f"Just so you know, I'm allergic to {food}.", "What am I allergic to?", f"You're allergic to {food}.",
         "food", (food,)),
        (f"My favorite color is {color}.", "What's my favorite color?", f"Your favorite color is {color}.", "color", (color,)),
        (f"I hid the {item} under the {place} near the {loc}.", f"Where did I hide the {item}?",
         f"You hid the {item} under the {place} near the {loc}.", "item", (place, loc)),
        (f"The password for the guild hall is '{w1} {w2}'.", "What is the guild hall password?",
         f"The guild hall password is '{w1} {w2}'.", "password", (f"{w1} {w2}",)),
        (f"I have a meeting with {n2} on {day} at {hour}.", f"When is my meeting with {n2}?",
         f"Your meeting with {n2} is on {day} at {hour}.", "meeting", (day, hour)),
        (f"My {rel}'s name is {n3}.", f"What is my {rel}'s name?", f"Your {rel}'s name is {n3}.", "relation", (n3,)),
        (f"I have a {pet} named {pet_name}.", "What pet do I have and what is its name?",
         f"You have a {pet} named {pet_name}.", "pet", (pet, pet_name)),
        (f"For the quest, I need to bring {count} wolf pelts to the elder in {city}.",
         "How many wolf pelts do I need for the quest?", f"You need to bring {count} wolf pelts.", "quest", (str(count),)),
    ]
    rng.shuffle(facts)
    return facts


def make_update(rng, key):
    """A later statement that overrides an earlier fact (tests overwriting old memories)."""
    if key == "color":
        c = rng.choice(COLORS)
        return f"Actually, I changed my mind, my favorite color is now {c}.", f"Your favorite color is now {c}.", (c,)
    if key == "code":
        c = _code(rng)
        return f"Update: the vault code was changed to {c}.", f"The vault code is now {c}.", (c,)
    if key == "password":
        w1, w2 = rng.sample(WORDS, 2)
        return (f"Heads up, the guild hall password changed to '{w1} {w2}'.",
                f"The guild hall password is now '{w1} {w2}'.", (f"{w1} {w2}",))
    return None


# ---------------------------------------------------------------------------------------------------------------
# Repetitive content generators. Each returns a callable producing the next short repetitive line.
# ---------------------------------------------------------------------------------------------------------------


def _npc_movement(rng):
    style = rng.randrange(5)
    x, y = rng.randint(0, 60), rng.randint(0, 60)
    dirs = [("north", 0, -1), ("south", 0, 1), ("east", 1, 0), ("west", -1, 0)]
    t = rng.randint(100, 9000)

    def tick():
        nonlocal x, y, t
        name, dx, dy = rng.choice(dirs)
        x, y, t = x + dx, y + dy, t + 1
        if style == 0:
            return f"move(dx={dx}, dy={dy})", f"[tick {t}] npc_pos=({x}, {y}) state=walking"
        if style == 1:
            return (f'{{"action": "move", "dir": "{name}", "steps": 1}}',
                    f'{{"event": "tick", "t": {t}, "npc_pos": [{x}, {y}], "player_dist": {rng.randint(3, 40)}}}')
        if style == 2:
            return f"MOVE {name[0].upper()} 1", f"TICK {t} POS {x},{y} OK"
        if style == 3:
            act = rng.choice(["walk_to", "step", "face", "walk_to"])
            return f"{act}({x}, {y})", f"tick={t} x={x} y={y} anim=walk"
        return rng.choice(["idle()", "look_around()", f"turn_{rng.choice(['left', 'right'])}()",
                           f"walk({name})"]), f"<tick {t}> position {x} {y}"

    return tick


def _telemetry(rng):
    temp, hum, co2 = rng.uniform(18, 25), rng.uniform(30, 60), rng.randint(400, 900)
    sensor = rng.choice(["livingroom", "greenhouse", "lab-3", "server-rack-2", "attic"])

    def tick():
        nonlocal temp, hum, co2
        temp += rng.uniform(-0.2, 0.2)
        hum += rng.uniform(-0.5, 0.5)
        co2 += rng.randint(-6, 6)
        line = f"sensor={sensor} temp={temp:.1f}C humidity={hum:.0f}% co2={co2}ppm status=ok"
        return "ack", line

    return tick


def _heartbeat(rng):
    svc = rng.choice(["auth-api", "payments", "edge-proxy", "worker-7", "db-replica", "scheduler"])
    sec = rng.randint(0, 3000)

    def tick():
        nonlocal sec
        sec += rng.choice([5, 5, 10])
        lvl = rng.choice(["INFO"] * 9 + ["DEBUG"])
        line = (f"2026-03-{rng.randint(1, 28):02d}T{(sec // 3600) % 24:02d}:{(sec // 60) % 60:02d}:{sec % 60:02d}Z "
                f"{lvl} {svc} heartbeat ok latency={rng.randint(3, 40)}ms queue={rng.randint(0, 4)}")
        return "OK", line

    return tick


def _chat_spam(rng):
    spam = ["lol", "LUL", "gg", "KEKW", "!points", "W", "pog", "first", "o7", "lmao", "Pog", "xD", "+1", "nice",
            "hi chat", ":)", "PogChamp", "L", "true", "same"]
    users = [f"user{rng.randint(10, 9999)}" for _ in range(12)]

    def tick():
        return "...", f"{rng.choice(users)}: {' '.join(rng.choice(spam) for _ in range(rng.randint(1, 3)))}"

    return tick


def _progress(rng):
    files = [f"chunk_{i:04d}.bin" for i in range(rng.randint(5, 40))]
    p = 0

    def tick():
        nonlocal p
        p = (p + rng.randint(1, 4)) % 101
        return "waiting", f"Downloading {rng.choice(files)} ... {p}% [{'#' * (p // 10)}{'.' * (10 - p // 10)}]"

    return tick


def _prices(rng):
    sym = rng.choice(["BTC", "ETH", "AAPL", "EURUSD", "GOLD", "TSLA"])
    px = rng.uniform(10, 70000)

    def tick():
        nonlocal px
        ch = rng.uniform(-0.002, 0.002)
        px *= 1 + ch
        return "noted", f"{sym} {px:.2f} {ch * 100:+.2f}% vol={rng.randint(10, 999)}"

    return tick


def _copy_paste(rng):
    lines = [
        "Thank you for contacting support. Your ticket has been received and will be processed shortly.",
        "This message was sent automatically. Please do not reply to this email.",
        "Reminder: the server will restart for maintenance. Save your work.",
        "You have unread notifications. Tap here to view them.",
    ]
    line = rng.choice(lines)

    def tick():
        return "ok", line

    return tick


FAMILIES = [_npc_movement, _npc_movement, _npc_movement, _telemetry, _heartbeat, _chat_spam, _progress, _prices,
            _copy_paste]

NPC_SYSTEMS = [
    "You are {npc}, a {job} NPC in the village of {town}. You control your own movement by replying with movement "
    "commands when you receive game ticks, and you talk naturally with players when they speak to you.",
    "You are the AI controller of {npc}, a companion character in an open-world game. Game engine ticks arrive as "
    "user messages; reply with a single action. When the player talks, answer in character.",
]
ASSIST_SYSTEMS = [
    "You are a helpful assistant.",
    "You are a helpful assistant that monitors data streams for the user and answers their questions.",
    "",
]
SMALL_TALK = [
    ("How's your day going?", "It's going well, thanks for asking! How can I help?"),
    ("Nice weather today, isn't it?", "It really is! Perfect for a walk."),
    ("Can you keep an eye on things for me for a while?", "Of course, I'll keep track of everything."),
    ("I'll be busy for a bit, keep doing your thing.", "Sounds good, I'll carry on."),
]


def generate_sample(tokenizer, rng: random.Random, target_tokens: int, return_values=False):
    """Builds one synthetic memory sample of roughly `target_tokens` tokens.

    Returns (ids, flags), plus the values the answer must contain when `return_values` is set.
    """
    facts = make_facts(rng)
    n_facts = rng.randint(2, 5)
    stated = facts[:n_facts]
    family = rng.choice(FAMILIES)(rng)
    npc_mode = family.__qualname__.startswith("_npc_movement") or rng.random() < 0.3
    as_turns = npc_mode or rng.random() < 0.5  # repetitive content as many short turns vs. one long pasted log

    messages = []
    if npc_mode:
        sysmsg = rng.choice(NPC_SYSTEMS).format(npc=rng.choice(FIRST), job=rng.choice(JOBS), town=rng.choice(CITIES))
    else:
        sysmsg = rng.choice(ASSIST_SYSTEMS)
    if sysmsg:
        messages.append({"role": "system", "content": sysmsg})

    # Facts, interleaved with a bit of natural small talk
    for statement, *_ in stated:
        if rng.random() < 0.3:
            q, a = rng.choice(SMALL_TALK)
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        prefix = "[Player]: " if npc_mode else ""
        messages += [{"role": "user", "content": prefix + statement},
                     {"role": "assistant", "content": rng.choice(["Got it, I'll remember that.", "Noted!",
                                                                    "Understood.", "Okay, I'll keep that in mind."])}]

    # Optional update of one fact, placed somewhere inside the filler
    update = None
    updatable = [f for f in stated if make_update(random.Random(0), f[3]) is not None]
    if updatable and rng.random() < 0.25:
        fact = rng.choice(updatable)
        update = (fact, make_update(rng, fact[3]))

    base_ids, _ = encode_segments(tokenizer, chat_segments(messages + [{"role": "user", "content": "x"}]))
    budget = max(target_tokens - len(base_ids) - 120, 512)
    filler_msgs, used, update_at, last_tick = [], 0, rng.uniform(0.2, 0.7), None
    if as_turns:
        while used < budget:
            batch = [family() for _ in range(32)]
            lengths = tokenizer([a + "\n" + t for a, t in batch], add_special_tokens=False)["input_ids"]
            for (action, tick), toks in zip(batch, lengths):
                if used >= budget:
                    break
                last_tick = tick
                filler_msgs += [{"role": "user", "content": tick, "flag": FLAG_FILLER},
                                {"role": "assistant", "content": action, "flag": FLAG_FILLER}]
                used += len(toks) + 9  # + chat-template tokens of the two turns
                if update and used > update_at * budget:
                    prefix = "[Player]: " if npc_mode else ""
                    filler_msgs += [{"role": "user", "content": prefix + update[1][0]},
                                    {"role": "assistant", "content": "Okay, updated."}]
                    update_at = 2.0
    else:
        chunks, cur = [], []
        while used < budget:
            lines = [family()[1] for _ in range(32)]
            for line, toks in zip(lines, tokenizer(lines, add_special_tokens=False)["input_ids"]):
                if used >= budget:
                    break
                cur.append(line)
                used += len(toks) + 1
                if update and used > update_at * budget:
                    chunks.append(("\n".join(cur), FLAG_FILLER))
                    chunks.append((update[1][0], 0))
                    cur, update_at = [], 2.0
        chunks.append(("\n".join(cur), FLAG_FILLER))
        intro = rng.choice(["Here's the log from the last few hours:", "Live feed:", "Stream output below.",
                            "Monitoring output:"])
        parts = [(intro + "\n", 0)]
        for k, (text, flag) in enumerate(chunks):
            parts.append((("" if k == 0 else "\n") + text, flag))
        filler_msgs.append({"role": "user", "parts": parts})
        filler_msgs.append({"role": "assistant", "content": rng.choice(["Received.", "Logged.", "OK."])})

    # Final question(s): mostly about early facts, sometimes about the latest repetitive state (still in window)
    asked = rng.sample(stated, rng.randint(1, min(2, len(stated))))
    answers, values = [], []
    for fact in asked:
        if update and fact is update[0]:
            answers.append(update[1][1])
            values.extend(update[1][2])
        else:
            answers.append(fact[2])
            values.extend(fact[4])
    question = " Also, ".join(f[1] if i == 0 else f[1][0].lower() + f[1][1:] for i, f in enumerate(asked))
    if last_tick is not None and rng.random() < 0.15:
        question += " And what did the most recent update say, exactly?"
        answers.append(f"The most recent update was: {last_tick}")
        values.append(last_tick)
    prefix = "[Player]: " if npc_mode else ""
    final = [{"role": "user", "content": prefix + question},
             {"role": "assistant", "content": " ".join(answers), "flag": FLAG_CE}]

    ids, flags = encode_segments(tokenizer, chat_segments(messages + filler_msgs + final))
    return (ids, flags, values) if return_values else (ids, flags)
