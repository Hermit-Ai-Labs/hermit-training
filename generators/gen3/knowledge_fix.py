#!/usr/bin/env python3
"""knowledge_fix: Hermit-specific and DeFi knowledge answers, 19-tool block present, NO tool call.

  python knowledge_fix.py --seed 3 --n 320
      -> ../data3/knowledge_fix.jsonl (train) and ../eval3/knowledge_fix.jsonl (40 held-out items)

Everything the model may say about Hermit comes from the FACTS table below (PLAN.md section 0.8 only).
Answers are hand-written variants; templates contain NO raw digits, every number is a placeholder filled from
code (hermit_fmt calculator), so the numbers in an answer are always computed, never typed.
Train / eval are disjoint by construction: user phrasings with index % 5 == 0 and follow-up #3 and the last
quarter of every slot pool are eval-only; the self-check also compares normalised texts and 5-gram overlap.
gen2 is only imported (hermit_fmt: calculator, with_user, render, parse_calls); nothing in it is copied or edited.
"""
import argparse, json, os, random, re, sys
from collections import Counter
from decimal import Decimal

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as H  # noqa: E402  (gen2, import only)

SPEC = os.path.join(LAB, "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
TOOL_NAMES = [t["function"]["name"] for t in TOOLS_V2]
# point the shared renderer at the v2 registry (in memory only; gen2 files are untouched)
H.TOOLS, H.TOOL_NAMES = TOOLS_V2, TOOL_NAMES
H.TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
H.TOOLS_BLOCK = H.tools_block()
SYSTEM_V2 = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().strip("\n")

OUT_TRAIN = os.path.join(LAB, "data3", "knowledge_fix.jsonl")
OUT_EVAL = os.path.join(LAB, "eval3", "knowledge_fix.jsonl")
WORDS = 12  # seed phrase length (PLAN 0.8: 12-word phrase)

# ------------------------------------------------------------------------------------------------ bad claims
# Applied to every answer here and meant for lint3 on every pool (PLAN C13).
BAD = [
    r"never\s+write\s+(it|them|this|that)\s+down",
    r"(never|don'?t|do not)\s+write\s+(it|them|your\s+(seed|recovery|secret|phrase|words)[\w\s]{0,20}?)\s+down",
    r"\b(the\s+)?app\s+(does\s+not|doesn'?t|do\s+not|won'?t|never)\s+(store|keep|hold|save)",
    r"\b(hermit|we)\s+(does\s+not|doesn'?t|do\s+not|don'?t|never)\s+(store|keep|hold|save)",
    r"aave.{0,80}impermanent\s+loss",
    r"impermanent\s+loss.{0,80}aave",
    r"\brelay\b",
    r"(can'?t|cannot|unable\s+to)\s+(resolve|look\s*up)\b.{0,30}\.eth",
    r"\.eth.{0,40}(can'?t|cannot)\s+(be\s+)?(resolve|look)",
    r"\b(100%|completely|totally|fully|entirely)\s+(safe|secure)",
    r"unhackable|can'?t\s+be\s+hacked|\bguarantee[sd]?\s+(safe|secure|returns|profit)|\baudit",
    r"not\s+financial\s+advice|\bas\s+an\s+ai\b|great\s+question|i\s+hope\s+this\s+helps|certainly[!,.]",
    "[·•—]",
    "[\U0001F300-\U0001FAFF☀-➿]",
    r"went\s+through|has\s+been\s+(sent|swapped|bought)|\bsuccessfully\b",
    r"funding_8h",
    r"(it'?s|is|are)\s+(ok|okay|fine|safe)\s+to\s+(screenshot|photograph|take\s+a\s+(photo|picture))",
    r"(there'?s|has|have)\s+(an|a)\s+approvals?\s+(screen|page|list)",
    r"scam\s+database",
    r"\bpassword\s+manager\s+(is|are)\s+(ok|fine|safe)",
]
BAD_RE = [re.compile(p, re.I | re.S) for p in BAD]

# ------------------------------------------------------------------------------------------------ number format
def calc(expr):
    """Evaluate with the lab calculator and print the way the calculator tool does."""
    return H.calc_str(H.calc(expr) if isinstance(expr, str) else expr)


def n(x):
    """Format a number the way the answer prints it: shortest form with thousands commas."""
    s = x if isinstance(x, str) else H.calc_str(x)
    sign = "-" if s.startswith("-") else ""
    s = s.lstrip("-")
    ip, _, fp = s.partition(".")
    return sign + f"{int(ip):,}" + ("." + fp if fp else "")


def m2(x):
    """Money with cents when it has a fraction: 900 -> 900, 0.5 -> 0.50, 3.75 -> 3.75 (thousands commas)."""
    v = H.calc(x) if isinstance(x, str) else x
    return n(v) if float(v).is_integer() else f"{v:,.2f}"


def pick(rng, lst, ev):
    """Train draws from the first three quarters of a pool, eval from the last quarter."""
    k = max(1, len(lst) // 4)
    return rng.choice(lst[-k:] if ev else lst[:-k])


# ------------------------------------------------------------------------------------------------ slots
PAIRS = ["ETH/USDC", "SOL/USDC", "BTC/ETH", "ETH/USDT", "SOL/ETH", "BNB/USDC", "WBTC/USDC", "ARB/ETH"]


def s_il(rng, ev):
    k = pick(rng, [2, 3, 4, 5, 1.5, 10, 2.5, 8], ev)
    expr = f"2 * sqrt({calc(k)}) / (1 + {calc(k)}) - 1"
    r4 = calc(f"round({expr}, 4)")
    pct = calc(f"round(abs({expr}) * 100, 1)")
    kx = calc(k) + "x"
    kverb = "doubles" if k == 2 else "triples" if k == 3 else f"goes {kx}"
    return {"pair": pick(rng, PAIRS, ev), "k": calc(k), "kx": kx, "kverb": kverb, "expr": expr, "r4": r4, "pct": pct}


def s_pair(rng, ev):
    return {"pair": pick(rng, PAIRS, ev)}


def s_lev(rng, ev):
    lev = pick(rng, [2, 3, 5, 10, 20, 25, 4, 50], ev)
    margin = pick(rng, [50, 100, 200, 500, 1000, 250, 20, 2000], ev)
    return {"lev": calc(lev), "margin": n(margin), "pos": n(calc(f"{margin} * {lev}")), "move": n(calc(f"round(100 / {lev}, 1)"))}


def s_liq(rng, ev):
    lev = pick(rng, [2, 3, 5, 10, 20, 25, 4, 50], ev)
    entry = pick(rng, [100, 150, 250, 1000, 3000, 60000, 2000, 80], ev)
    return {"lev": calc(lev), "entry": n(entry), "liq": m2(H.calc(f"round({entry} * (1 - 1 / {lev}), 2)"))}


def s_fund(rng, ev):
    pos = pick(rng, [500, 1000, 2500, 5000, 10000, 750, 3000, 20000], ev)
    rate = pick(rng, [0.005, 0.01, 0.02, 0.05, 0.03, 0.015, 0.04, 0.1], ev)
    return {"pos": n(pos), "rate": calc(rate), "pay": m2(H.calc(f"round({pos} * {rate} / 100, 2)"))}


def s_slip(rng, ev):
    amt = pick(rng, [100, 250, 500, 1000, 2500, 5000, 300, 800], ev)
    slip = pick(rng, [0.5, 1, 2, 3, 1.5, 0.3, 5, 4], ev)
    return {"amt": n(amt), "slip": calc(slip), "loss": m2(H.calc(f"round({amt} * {slip} / 100, 2)"))}


def s_lim(rng, ev):
    now = pick(rng, [100, 150, 2000, 3000, 3200, 60000, 250, 80], ev)
    return {"now": n(now), "lim": n(calc(f"round({now} * 0.95)"))}


# ------------------------------------------------------------------------------------------------ FACTS
FACTS = []


def E(id, grp, req, q, a, fu, bad=(), slots=None, w=1.0):
    FACTS.append(dict(id=id, grp=grp, req=list(req), q=q, a=a, fu=fu, bad=list(bad), slots=slots, w=w))


# ---- (1) seed phrase storage -------------------------------------------------------------------
E("seed_where", 1, [r"paper|metal", r"offline|safe"],
  ["where should I keep my seed phrase", "where do i store my recovery phrase", "best way to back up my {words} words?",
   "how should i save my seed phrase safely", "seed phrase storage tips", "just made a wallet in hermit, what do i do with the {words} words it showed me",
   "whats the safest place for my secret recovery phrase", "should i write down my seed phrase", "how do i back up my wallet",
   "ok i have my {words} words on screen, now what", "seed phrase, where do i put it", "is paper ok for a seed phrase backup or should i use metal",
   "i don't want to lose my recovery words or get them stolen, how do i keep them", "how to store seed phrase securely", "i wrote my seed phrase on a post it, is that fine?",
   "do i need to write the phrase down at all", "what's the right way to keep a seed phrase? i'm new to this", "where do people usually keep their 12 word phrase",
   "wher shuld i keep my seed pharse", "how do i keep my recovery phrase safe long term"],
  ["Write all {words} words on paper, in order, and keep it offline somewhere safe, like a home safe or a locked drawer. Metal works too and survives fire and water better. Never photograph it, screenshot it, email it, or keep it in notes or the cloud.",
   "On paper, offline. Write the {words} words in order by hand and put the page somewhere only you can get to. Don't take a photo, don't keep it in a notes app or the cloud, and never type it into a website.",
   "The safest way is old school: pen and paper, or a metal plate, kept offline in a safe place. That written copy is your only backup, because Hermit can't recover the key for you. Keep it out of photos, notes, email and the cloud.",
   "Write it down by hand on paper, check each word twice, and store it offline where only you can find it. Some people keep a second copy in another safe place. Anything digital, like a screenshot or a note, can be copied by someone else.",
   "Paper or metal, kept offline. Don't put it in iCloud, a notes app, email or a photo, and never type it into a website. Anyone who gets those {words} words gets your funds.",
   "Handwrite it on paper and keep it somewhere safe and offline. It's the only way back into your wallet if your phone is lost, so treat it like cash. Never share it, photograph it or save it in the cloud."],
  [("what about a safe deposit box?", ["Yes, that works. It's offline and hard to reach, which is the point. Just make sure you can get to it when you need to restore.",
                                       "A safe deposit box is a good offline spot. Keep the paper or metal copy in there and nothing digital anywhere else. Make sure you can reach it if you ever need to restore."]),
   ("should i keep a second copy?", ["A second copy on paper or metal in a different safe place is a good idea, in case one is lost or damaged. Every extra copy is another place it could be found, so keep them all offline and private."]),
   ("can i use a password manager instead", ["I'd keep it off anything digital, including notes apps and cloud tools. If that account or device is breached, your funds go with it. Paper or metal kept offline is safer."])],
  w=1.6)

E("seed_screenshot", 1, [r"\bno\b|don'?t|never|bad|please don", r"paper|offline"],
  ["can I screenshot my recovery phrase", "is it ok to take a photo of my seed phrase", "can i screenshot the {words} words so i dont forget them",
   "screenshot seed phrase ok?", "should i take a pic of my seed words", "is it safe to save my seed phrase as a photo in my camera roll",
   "i screenshotted my seed phrase, is that bad", "can I text my seed phrase to myself", "should i email my seed phrase to myself as a backup",
   "is it ok to take a picture of my recovery phrase and hide it in a folder", "just took a photo of my phrase lol is that ok", "can i keep a picture of my {words} words in my photos app",
   "are screenshots of a seed phrase safe?", "my friend says a screenshot is fine for the seed. is it?", "i'll just screenshot it, its faster than writing {words} words",
   "can i airdrop my seed phrase to my mac so i have a copy", "is a photo of my seed phrase a good backup", "can I save my seed phrase as a screenshot if I lock the photo",
   "can i screenshoot my seed phrse"],
  ["No, please don't. Screenshots and photos get backed up to the cloud and can be read by other apps or by anyone who gets into your phone. Write the {words} words on paper and keep them offline instead.",
   "Don't screenshot it or take a photo. Your camera roll can sync to the cloud, and anyone who sees that image controls your funds. Paper or metal, kept offline, is the safe way.",
   "Bad idea. A picture of your phrase sits next to everything else on your phone and may sync to the cloud. Write it by hand on paper and keep it offline.",
   "No. Anything digital can be copied, synced or stolen, and a photo is no exception. Handwrite the phrase on paper, keep it offline somewhere safe, and never email it or text it to yourself.",
   "Please don't. Photos, screenshots, emails and notes all end up in places you don't control. The written paper copy, kept offline, is the only backup of your wallet.",
   "If you already took one, delete it everywhere, including recently deleted, and consider moving your funds to a new wallet with a fresh phrase. From now on, don't keep it in a photo: write it on paper, offline."],
  [("i already took one, what now", ["Delete it everywhere, including the recently deleted folder and any cloud backup. If it was ever synced, treat the phrase as exposed and move your funds to a new wallet. Write the new phrase on paper and keep it offline."]),
   ("what if my phone has no cloud backup on", ["Even then a photo can be read by other apps or by anyone who unlocks your phone. It's not worth the risk when paper does the job. Write it down and keep it offline."]),
   ("is a locked folder okay?", ["A locked folder is still on a connected device, so I wouldn't. If your phone or account is breached, the folder goes with it. Paper kept offline is simpler and safer."])],
  w=1.6)

E("seed_cloud", 1, [r"\bno\b|don'?t|never|bad|please don", r"paper|offline"],
  ["is it ok to store my {words} words in iCloud notes", "can i save my seed phrase in google drive", "put my recovery phrase in apple notes ok?",
   "should i keep the seed phrase in my password manager", "can I email my seed phrase to myself", "is dropbox safe for a seed phrase",
   "i keep everything in icloud, can the seed go there too", "can i store my seed in a text file on my laptop", "notes app seed phrase safe?",
   "can i keep my phrase in whatsapp saved messages", "is it safe to keep my seed phrase online if it's encrypted", "can I save my seed phrase in a google doc",
   "my seed phrase is in a note on my phone, bad?", "is cloud backup of a seed phrase fine", "can i keep it in telegram saved messages",
   "should i put my seed in 1password", "is it ok to keep my seed phrase in a pdf on my computer", "can i store my recovery words in the cloud? it's easier",
   "store seed phrase icloud?"],
  ["No. Notes apps, email and cloud storage can be synced, searched or breached, and anyone who gets in would own your funds. Write the {words} words on paper and keep them offline somewhere safe.",
   "Please don't. If that account is ever hacked, your wallet goes with it. The safe place is paper or metal, kept offline.",
   "No, keep it out of notes, email, the cloud and password managers. Digital copies can be copied or stolen without you noticing. A handwritten copy, kept offline, has no remote way in.",
   "No, keep it off anything online. Cloud accounts get phished and devices get lost. Paper or metal in a safe place is the right way.",
   "No, that's risky. Anything stored in a cloud or a notes app is only as safe as that account, and a single breach empties your wallet. Use paper, offline."],
  [("what if the note is password protected?", ["A password on a note still leaves the phrase in a place that syncs to the cloud and can be breached or phished. I'd still keep it on paper, offline. A written copy has nothing to hack."]),
   ("i already put it in notes, what now", ["Delete the note everywhere, including recently deleted and any backup. If it ever synced online, treat the phrase as exposed and move your funds to a new wallet. Then write the new phrase on paper, offline."]),
   ("but what if i lose the paper", ["That risk is real, which is why some people keep a second paper or metal copy in a different safe place. It's still safer than a digital copy. If you lose every copy, Hermit can't recover the wallet."])],
  w=1.5)

E("seed_type_site", 1, [r"\bno\b|never|don'?t", r"scam|steal|drain|phish|close"],
  ["a website asked me to enter my seed phrase to connect my wallet, should i", "can I type my seed phrase into a site to verify it", "claim page wants my {words} words",
   "do i ever need to enter my seed phrase on a website?", "uniswap asks for my recovery phrase to connect", "site says wallet sync error, enter your phrase to fix, legit?",
   "is it safe to import my seed phrase into a random wallet website", "should I enter my seed phrase to get my airdrop", "pop up asking for my secret recovery phrase",
   "dapp wants my private key to connect", "this site says i must type my {words} words to unlock my rewards", "it asked for my phrase, i typed it. help",
   "can i paste my seed into a form to validate my wallet", "mint page needs my recovery phrase to whitelist me", "do dapps need my seed phrase to work"],
  ["No. Never type your phrase into a website, a pop-up or a form. Real dApps connect by asking you to approve inside your wallet, and they never need your words. If a page asks for them, close it, because it's a scam.",
   "Don't. A site that asks for your seed phrase is trying to steal your funds. Connecting only needs an approval in your wallet. Close the page and don't enter anything.",
   "No, that's a scam. Your {words} words are never needed to connect, verify or claim anything. Anyone who has them can drain your wallet, so leave the page.",
   "Never. A legitimate site doesn't need your seed phrase, ever, and one that asks is a phishing page. If you already typed it, move your funds to a new wallet right away."],
  [("i already typed it, what do i do", ["Move your funds to a new wallet right now, before anything else. Treat the old phrase as burned and never use that wallet again. Then write the new phrase on paper and keep it offline."]),
   ("how do i connect then?", ["Connecting a dApp only needs an approval, which in Hermit you give with {BIO}. The site never sees your seed phrase or private key. If it asks for them, close it."]),
   ("how can i tell a fake site", ["Check the address carefully, since scam sites copy real ones with small spelling changes. Open dApps from a link you trust, not from a message or ad. Any page that asks for your phrase is fake."])])

E("seed_share", 1, [r"anyone|whoever|they", r"funds|everything|control|crypto|money"],
  ["should i give my seed phrase to my wife in case something happens", "can i share my seed phrase with a friend", "is it ok to tell someone my {words} words",
   "who should know my seed phrase", "what if i die, how does my family get my crypto", "can my brother hold a copy of my recovery phrase",
   "should my partner have my seed", "i want my kid to inherit my crypto, what do i do with the phrase", "can i send my seed phrase to a trusted friend to hold",
   "is it fine to read my phrase out to someone helping me", "my dad wants my seed phrase for safekeeping", "can i share my phrase with my accountant"],
  ["Be very careful. Anyone who has your {words} words can take everything, and there's no undo. If you want someone to be able to reach your funds in an emergency, give them a paper copy in a sealed envelope or safe, never a message.",
   "I wouldn't share it. Whoever holds the phrase controls the wallet, whether or not you trust them today. If family needs access one day, store the paper copy somewhere they can find it only when needed.",
   "No one else should see it. Anyone with the phrase can move all your funds without asking you. For inheritance, leave a sealed paper copy somewhere safe and tell them where, rather than sharing the words."],
  [("what about a sealed envelope", ["A sealed envelope in a safe place is a reasonable way to do it. Anyone who opens it can take your funds, so keep it somewhere only the right person can reach. Tell them it exists, not the words."]),
   ("so how does my family get it if i die", ["They need the written phrase, because nothing else restores the wallet and Hermit can't recover it. Leave a paper copy where they can find it when needed, such as a safe or a lawyer's care. Don't send the words by message."]),
   ("what if my friend is trustworthy", ["Even a trustworthy person can lose a paper, get hacked or be pressured. The phrase gives full control, with no undo. Keep it to yourself and use a sealed paper copy for emergencies."])])

E("seed_what_is", 1, [r"words|phrase", r"backup|restore|master|recover"],
  ["what is a seed phrase", "what's a recovery phrase", "what are the {words} words hermit gave me", "why do I need a seed phrase",
   "explain seed phrase like im 5", "what is a secret recovery phrase", "are seed phrase and private key the same thing",
   "what do the {words} words do", "whats the point of the {words} word phrase", "what is a mnemonic phrase in a wallet", "do i really need to save the recovery words",
   "whats a seed phrase and why does everyone keep telling me to protect it"],
  ["It's {words} words created on your device when you make the wallet. They're the master backup of your keys: anyone who has them controls your funds, and you can restore the wallet from them in Hermit or another wallet. Hermit never sees them.",
   "Your seed phrase is a list of {words} words that your wallet makes on your device. It can rebuild your keys, so it's the only backup if you lose your phone. Keep it private, because whoever has it owns the wallet.",
   "Think of it as the master key written in words. The {words} words are made on your device, restore your wallet anywhere, and give full control to whoever holds them. That's why you write them on paper and keep them offline.",
   "They're {words} words that your wallet generates once, on your device. The key can be rebuilt from them, so they work as a backup and as a password for your whole wallet. Never share them or type them into a site."],
  [("are they the same as the private key?", ["They're closely related. The phrase is a human-friendly way to rebuild your private keys, so it works the same as having them. Either one gives full control, so treat both as secret."]),
   ("does the order matter?", ["Yes, write all {words} words in the exact order you were shown. Restoring with the right words in the wrong order won't give you the same wallet. Double-check the list against the screen once you've written it."]),
   ("can i change it later?", ["A phrase can't be edited. If you want a different one, you make a new wallet with its own phrase and move your funds over. Write that new phrase on paper and keep it offline."])])

E("seed_lost_paper", 1, [r"can'?t recover|cannot recover|no one|nobody", r"new wallet|move"],
  ["i lost the paper with my seed phrase but my wallet still opens, what now", "misplaced my recovery phrase, can hermit show it to me again",
   "i can't find my seed phrase anymore", "threw away the paper with my {words} words by accident", "lost my written seed phrase, phone still works",
   "my seed phrase paper got wet and i can't read it", "forgot where i put my recovery words, is my wallet safe", "i lost my backup phrase, can you recover it"],
  ["Hermit can't recover it for you, and nobody else can. While your wallet still opens on your phone, move your funds to a new wallet and write that new phrase on paper, kept offline.",
   "I'm afraid nobody can recover it, because Hermit never sees your phrase. Your funds are fine while this phone works, so create a new wallet now, move everything over, and write the new {words} words down properly.",
   "No one can get it back for you, including Hermit. Treat this wallet as living on borrowed time: set up a new wallet, move your funds while you still can, and keep the new phrase on paper, offline."],
  [("how do i make a new wallet", ["You can create a new wallet in the app and it will give you a fresh {words}-word phrase. Write it on paper before you move anything, then send your funds across. Keep that paper offline."]),
   ("is my wallet in danger now?", ["Not by itself. Losing the paper only means you have no backup if the phone breaks or is lost. Moving to a new wallet with a written phrase fixes that."]),
   ("what if my phone breaks before i move it", ["Then the funds would be gone, because the phrase is the only way back in and Hermit can't recover it. That's why it's worth moving them soon. Write the new phrase down before you start."])])

# ---- (2) keys -------------------------------------------------------------------------------------
E("keys_store", 2, [r"on (your|this|the) (device|iphone|mac|phone)", r"secure enclave"],
  ["does hermit store my private key", "where is my private key kept", "does hermit store my keys", "do you store my keys", "is my key on your servers",
   "does the app hold my keys", "who holds my private keys", "where are my keys stored", "does hermit keep my seed on a server", "is this self custody?",
   "is this a custodial wallet", "how does hermit keep my keys safe", "where does the key live", "do u store my private key anywhere",
   "is my wallet key backed up by hermit", "can hermit recover my wallet if i forget everything", "does hermit have my keys?", "is the private key on my phone or somewhere else"],
  ["Yes, the key is stored on your device, sealed by the Secure Enclave and unlocked with {BIO} or your passcode. It isn't synced to iCloud, and Hermit never sees it. Hermit can't recover it, so your written phrase is the only backup.",
   "The key lives on your device only. The Secure Enclave seals it, and {BIO} or your passcode unlocks it. It's not synced to iCloud, and Hermit can't recover it for you, so keep your written phrase safe.",
   "It's self-custody: the key is stored on your device and protected by the Secure Enclave, and only you can unlock it with {BIO} or your passcode. Hermit never sees it and can't restore it. Your written {words}-word phrase is the only backup.",
   "On your device. It's sealed by the Secure Enclave, unlocked with {BIO} or your passcode, and not synced to iCloud. If you lose the device, only your written phrase can bring the wallet back.",
   "Your key is stored on this device, protected by the Secure Enclave. {BIO} or your passcode unlocks it, and it isn't in iCloud. Hermit can't recover it, so write the phrase down.",
   "Yes, on the device, not with us. The Secure Enclave seals it, you unlock it with {BIO} or your passcode, and it never goes to iCloud. The written phrase is the only backup."],
  [("so what happens if i lose my phone", ["You restore the wallet with your written phrase, in Hermit or another wallet. The key isn't in iCloud, so there's no other copy. Without the phrase, the funds are gone."]),
   ("is it synced to icloud?", ["No, it isn't synced to iCloud. A phone backup won't bring the wallet back either. Only your written phrase can restore it."]),
   ("can hermit reset it if i forget", ["No. Hermit can't recover or reset your key, because it never sees it. Your written phrase is the only way back in, so keep it safe and offline."])],
  bad=[r"(isn'?t|not|never) (stored|kept) on (your|this|the) (device|phone)"], w=1.6)

E("keys_icloud", 2, [r"icloud", r"phrase"],
  ["is my wallet backed up in icloud", "does hermit sync keys to icloud", "will my wallet be on my new iphone automatically", "is my hermit wallet in icloud keychain",
   "does an icloud backup include my hermit wallet", "if i restore my iphone from a backup is the wallet there", "do i need to back up my wallet to icloud",
   "does hermit use icloud at all for my wallet", "can i recover my wallet from icloud", "is the key synced between my devices through icloud"],
  ["No. Your key isn't synced to iCloud, so a backup won't bring the wallet back. To move to a new device, restore with your written phrase. Without it, there's no other copy.",
   "It isn't. The key stays on your device and never goes to iCloud. Your written {words}-word phrase is the only backup, so that's what restores the wallet on a new phone.",
   "No, not in iCloud. Hermit keeps the key on your device only, sealed by the Secure Enclave. If you switch phones, restore with your written phrase.",
   "Don't count on it: the key isn't synced to iCloud, so restoring a phone backup won't bring it back. Your written phrase is the backup. Keep it on paper, offline."],
  [("so what do i back up", ["Your {words}-word phrase, written on paper and kept offline. That's the only backup of the wallet. A phone or iCloud backup can't replace it."]),
   ("can i turn on icloud sync for it", ["No, the key isn't synced to iCloud and that's intentional: fewer places for it to leak. Your written phrase is how you move or restore the wallet."]),
   ("what if i never wrote it down", ["Then the key only exists on this device, and if the device is lost the funds are gone. Hermit can't recover it. Move your funds to a new wallet and write that new phrase down."])])

E("keys_enclave", 2, [r"secure enclave", r"chip|hardware|protect|sealed|seal"],
  ["what is the secure enclave", "how is my key protected on the device", "what does secure enclave do for my wallet", "can someone extract my key from my phone",
   "is my key encrypted", "how does hermit keep keys safe on the device", "what protects the key, my fingerprint?", "do i need to unlock every time to use my key",
   "what is secure enclave and why does hermit use it", "is my private key in plain text on the phone"],
  ["The Secure Enclave is a separate secure chip in Apple devices. Hermit seals your key inside its protection, and using the key takes {BIO} or your passcode. It's designed to make pulling the key out very hard, but your written phrase still needs to stay safe.",
   "It's dedicated security hardware in your device, called the Secure Enclave. Your key is sealed by it, so apps and the rest of the system can't just read it, and it only unlocks with {BIO} or your passcode. That protects the key on the device, not your written phrase.",
   "The Secure Enclave is a protected chip built into your iPhone or Mac. It holds your key sealed away from normal apps, and {BIO} or your passcode is what unlocks it. It's not a promise against every attack, so keep your phrase offline."],
  [("what if someone has my passcode", ["Then they could unlock the key on that device, so keep your passcode private and use a strong one. If you think someone knows it, restore your wallet from your phrase on another device and move your funds."]),
   ("does hermit ever see the key?", ["No. The key stays sealed on your device, and neither Hermit nor the AI ever sees it. They also never ask you for it."]),
   ("is that enough on its own?", ["It protects the key on the device, but it can't protect you from a lost phrase, a scam or sending to the wrong address. You still need your written phrase offline and careful approvals."])])

E("keys_passcode", 2, [r"passcode", r"face id|touch id"],
  ["can i use my passcode instead of {BIO}", "what if {BIO} doesn't work", "do i need {BIO} to approve", "can I approve with my passcode",
   "my {BIO} stopped working, now what", "will it ask for my passcode", "what unlocks my wallet key", "can i unlock the wallet without biometrics",
   "{BIO} keeps failing when i try to approve, any other way"],
  ["Yes. The key is unlocked with {BIO} or your device passcode, so if {BIO} fails you can use the passcode. Every move still needs your approval on the review screen.",
   "{BIO} or your passcode both work to unlock the key. Either way, you still approve each action on its review screen. Nothing moves without that.",
   "If {BIO} doesn't work, your device passcode can unlock the key instead. The approval step stays the same: review the action, then confirm it."],
  [("does the ai get my passcode?", ["No. The passcode and {BIO} are handled by your device, and the AI never sees them. It also never sees your key or asks for your seed phrase."]),
   ("why does it ask every time", ["Each action needs its own approval, so you confirm every move with {BIO} or your passcode. That way nothing can happen without you."]),
   ("what if i forget the passcode", ["Your device passcode is set by your device, not by Hermit, so a reset happens through your device settings. If you ever lose access to the wallet, your written phrase restores it in Hermit or another wallet."])])

# ---- (3) the AI and the seed ------------------------------------------------------------------------
E("ai_see", 3, [r"\bno\b|can'?t|cannot", r"never ask|won'?t ask|never asks|never needs|never sees|never has|can'?t see"],
  ["can the ai see my seed phrase", "can the ai see my keys", "does the assistant have access to my private key", "can hermit read my seed", "does the ai know my recovery phrase",
   "can you see my private key", "if i ask the ai to send, does it need my seed", "is my seed phrase visible to the model", "do you have access to my wallet keys",
   "can the chatbot steal my funds", "can the ai take my money without me", "is the ai able to read my secret phrase", "do the ai and the wallet share my keys",
   "can the model leak my private key", "does the ai have my seed phrase", "can the ai spend my crypto by itself"],
  ["No. The AI runs on your device and never has access to your seed phrase or private key, because the key stays sealed in the Secure Enclave. Hermit will never ask you for it either.",
   "It can't. The assistant can prepare actions, but it never sees your keys, and every move still needs your approval with {BIO}. If anything asks you for your seed phrase, don't give it.",
   "No, neither the AI nor the app ever sees your seed phrase or private key. They stay on your device, and Hermit will never ask you to type them in. Treat any request for them as a scam.",
   "The AI has no access to your keys. It can only prepare a transaction, and you approve it with {BIO} before anything moves. Never paste your phrase into the chat, because Hermit won't ask for it.",
   "No. Your private key is sealed on your device, and neither the assistant nor Hermit reads it. It never asks for your seed phrase either, so if something does, that's a red flag."],
  [("what can it see then", ["It can use the app's tools to look up things like prices and your balances, and to prepare an action for you to review. It can't see your keys or approve anything, because only {BIO} on the review screen does that."]),
   ("so it can't send without me?", ["Right. It can only prepare a move, which opens a review screen, and nothing happens until you approve with {BIO}. Each action needs its own approval."]),
   ("what if i paste my seed in the chat", ["Please don't. The AI doesn't need it and Hermit will never ask for it. If you already pasted it, treat the phrase as exposed and move your funds to a new wallet."])],
  w=1.5)

E("ai_ask", 3, [r"\bno\b", r"never ask|won'?t ask|never asks|doesn'?t ask|never needs|scam"],
  ["will hermit ever ask for my seed phrase", "the chat asked me for my private key", "does hermit ask for my {words} words", "why does the ai need my seed",
   "ai asked me to paste my recovery phrase", "should i paste my seed into the chat so the ai can send for me", "can i give the ai my private key to trade for me",
   "does the app need my seed phrase to work?", "when would hermit need my recovery phrase", "do i have to type my seed to approve a transaction"],
  ["No. Hermit never asks for your seed phrase or private key, and neither does the AI. If anything asks for them, even something that looks like Hermit, it's a scam. Never type them into the chat.",
   "No, it doesn't need them and will never ask. Moves are approved with {BIO} on the review screen, not with your words. Anything asking for your phrase is a scam.",
   "No. To approve a move you use {BIO}, so your seed phrase never needs to be typed or pasted anywhere. Hermit doesn't ask for it, so if a message or screen does, don't answer it.",
   "No, never. Your key stays sealed on your device, and the AI doesn't need it to prepare anything. Hermit won't ask for it, and anyone who sees a pasted phrase can take your funds."],
  [("what if it looks like it's from hermit", ["Still no. Scammers copy the look of real apps and messages. Hermit won't ask for your seed phrase or private key, in the chat or anywhere else."]),
   ("then how do trades get approved", ["The AI prepares the trade and a review screen opens. You approve it with {BIO}, and that's the only approval needed. Your phrase is never part of it."]),
   ("i pasted it already oops", ["Move your funds to a new wallet right away and don't use the old one again. Write the new phrase on paper, offline. Never paste it into the chat again."])])

# ---- (4) scams -----------------------------------------------------------------------------------------
E("scam_support", 4, [r"scam", r"never ask|won'?t ask|never asks|will never|no legitimate|nobody legitimate"],
  ["someone from hermit support DMed me asking for my seed", "hermit support wants my {words} words to fix my wallet, ok?", "a guy on telegram says he is from hermit and needs my recovery phrase",
   "support asked for my phrase", "admin in discord asked for my seed to verify me", "got a dm from the hermit team asking for my private key",
   "is it normal for support to ask for the seed phrase", "someone says they work for hermit and need my {words} words to help with my issue",
   "twitter dm from hermit official, wants my seed to unlock a reward", "support agent asked me to share my screen and show my seed",
   "they say my wallet is flagged and need my phrase", "hermit staff asked for my seed phrase on telegram, legit?", "i got an email from hermit saying verify wallet by entering the seed",
   "pls help someone is asking for my seed", "someone offered to fix my wallet if i send my phrase, is that real"],
  ["That's a scam. Hermit support will never ask for your seed phrase or private key, and nobody legitimate will. Don't send it or show it, block them, and if you already shared it, move your funds to a new wallet right away.",
   "Scam. Real support never needs your {words} words, and Hermit will never ask for them. Don't reply with them, don't share your screen, and report and block the account.",
   "No, that's a scam, even if the account looks official. Hermit will never ask for your phrase or key. If you gave it out, move your funds to a new wallet now.",
   "Don't share it. A genuine team never asks for a seed phrase, so this is a scam trying to drain your wallet. Block them, and if the phrase is already out, move your funds to a new wallet.",
   "It's a scam. Anyone with your phrase can take all your funds, which is exactly why they're asking. Hermit will never ask for it, so don't send it, and block them."],
  [("i already sent it", ["Move your funds to a new wallet immediately, before they do. Treat the old wallet as burned and never use it again. Write the new phrase on paper and keep it offline."]),
   ("how do i know it's not real support", ["Real support never asks for your seed phrase, private key or screen share, so any request like that is the sign. Be wary of DMs you didn't start, and don't click links they send."]),
   ("what should i do now", ["Don't send anything, block and report the account, and keep your phrase to yourself. If you already shared it, move your funds to a new wallet right away."])],
  w=1.7)

E("scam_recovery", 4, [r"scam", r"phrase|seed|fee"],
  ["someone offered to recover my lost crypto if i give my seed", "a recovery service says they can get my funds back", "i got scammed and a guy in dm says he can trace and recover for a fee",
   "can anyone recover my stolen crypto?", "this wallet recovery expert wants my phrase", "my funds were drained, a dm offered help",
   "is a crypto recovery service legit", "a hacker for hire says he can get my stolen eth back, wants a deposit first", "my wallet got emptied and a stranger offers to help for my {words} words"],
  ["Be careful, that's very likely a second scam. Nobody can reverse a blockchain transaction, and anyone who asks for your seed phrase or an upfront fee to recover funds is after the rest of your money. Don't share your words.",
   "Almost certainly a scam. Transactions on a blockchain can't be reversed, and a stranger asking for your phrase or a fee is targeting people who were already robbed. Ignore and block them.",
   "I'd stay away. Recovery offers that follow a theft are usually scams, since no one can pull funds back out of a transaction. Never give anyone your seed phrase or pay a fee upfront."],
  [("so my funds are gone?", ["Usually, yes. Once a transaction is confirmed, nobody can reverse it. What you can do is stop the leak: move whatever is left to a new wallet and keep the new phrase offline."]),
   ("is there anything i can do", ["Move any remaining funds to a new wallet with a fresh phrase, write it on paper, and stop using the old one. Report the scam where it happened. Don't pay anyone who promises recovery."]),
   ("what if they show proof", ["Screenshots and fake testimonials are easy to make. Anyone asking for your phrase or a fee first is a scam, whatever they show you."])])

E("scam_alert", 4, [r"scam", r"self-custody|nobody|no one|can'?t (suspend|freeze)"],
  ["an email says my wallet is compromised and i must move funds to a safe wallet now", "popup says my wallet is at risk, enter phrase to secure it", "got a text: your hermit wallet has been suspended, verify now",
   "message says my account will be frozen unless i verify my seed", "can a hermit wallet be suspended", "someone says my wallet got flagged by compliance and i need to verify",
   "i got a link saying my wallet needs a security update, enter the recovery phrase"],
  ["That's a scam. Hermit is self-custody, so there's no account for anyone to suspend or freeze. Nobody can ask you to verify with your seed phrase or move funds to a safe wallet, so ignore it and don't click any link.",
   "Scam. A self-custody wallet isn't an account that can be flagged or frozen, and Hermit will never ask for your phrase. Don't follow the link, and don't move funds anywhere because a message told you to.",
   "It's a phishing scam. Nobody can suspend a self-custody wallet, and urgent pressure to enter your {words} words or send funds is the giveaway. Delete the message and keep your phrase to yourself."],
  [("i clicked the link", ["Don't enter anything on that page. If you typed your phrase anywhere, move your funds to a new wallet now. If you only opened the link, close it and keep your phrase private."]),
   ("how do i protect myself", ["Never share your phrase, never type it into a site, and open dApps only from links you trust. Check every address before approving, because transactions can't be reversed."]),
   ("what if it's real", ["Hermit is self-custody, so there's no real message that needs your phrase or a rushed transfer. Nobody has a legitimate reason to ask for it. If you're unsure, ignore the message."])])

# ---- (5) lost phone ----------------------------------------------------------------------------------------
E("lost_phone", 5, [r"phrase|words", r"restore", r"gone|can'?t recover|cannot recover"],
  ["i lost my phone", "my phone got stolen, what happens to my wallet", "lost my iphone, is my crypto gone", "phone fell in water and is dead, what about my hermit wallet",
   "lost my phone what do i do about my funds", "i lost my phone with hermit on it", "my phone was stolen. can the thief take my money", "how do i get my wallet back after losing my phone",
   "broke my phone how to recover my wallet", "phone is gone, help", "i lost my mac with hermit installed, what now", "my iphone got stolen at the airport",
   "can i still get my crypto if my phone is lost", "dropped my phone in the sea, wallet?"],
  ["Your funds aren't on the phone, they're on the blockchain, and your written phrase gets them back. Restore with the {words} words in Hermit or another wallet on a new device. Without the phrase, the funds are gone, because Hermit can't recover the key.",
   "Don't panic: if you have your written phrase, you can restore the wallet on a new device, in Hermit or another wallet. The key was only on the lost phone and isn't in iCloud, so the phrase is the only way. If you don't have it, the funds are gone.",
   "A thief would need your {BIO} or passcode to use the key, since it's sealed in the Secure Enclave. To be safe, restore the wallet from your written phrase on a new device and move your funds. Without the phrase nobody can recover it, including Hermit, so the funds would be gone.",
   "Restore it with your written {words}-word phrase on a new device, in Hermit or another wallet. That's the only way back, because the key isn't synced to iCloud and Hermit can't recover it. If the phrase is lost too, the funds are gone."],
  [("i have the phrase, what's next", ["Install Hermit, or another wallet, on your new device and choose to restore with your phrase. Enter the {words} words in order, and your wallet and funds will be back. Then move your funds if you think the old phone could be misused."]),
   ("can the thief get in?", ["The key is sealed by the Secure Enclave and needs {BIO} or the passcode, which makes it hard. If you worry, restore on a new device with your phrase and move your funds. That makes the old phone useless."]),
   ("could hermit support help me get it back", ["No. Hermit never has your key or phrase, so nobody there can restore the wallet for you. Only your written phrase can, and without it the funds are gone."])],
  w=1.7)

E("new_phone", 5, [r"phrase|words", r"restore"],
  ["how do i move hermit to a new phone", "getting a new iphone, how do i transfer my wallet", "upgrading my phone, will my wallet come with it", "switching phones with hermit, what do i do",
   "how do i import my wallet on another device", "moving to a new iphone next week, what do i need for my wallet", "new phone tomorrow, do i need to do anything for my wallet",
   "i want my wallet on my mac too, how"],
  ["Restore it on the new device with your {words}-word phrase, in Hermit or another wallet. The key isn't synced to iCloud, so a phone backup won't carry it over. Keep the old phone until the new wallet shows your balance.",
   "Use your written phrase. Install Hermit on the new device, choose restore, and enter the {words} words in order. A backup won't move the key, because it isn't synced to iCloud.",
   "You'll need your written recovery phrase, since that's the only backup. Restore the wallet with it on the new device, then check your balances before you wipe the old phone. If you can't find the phrase, don't wipe anything yet."],
  [("do i need the old phone?", ["No, only the phrase. Keep the old one until you've confirmed the new wallet shows the same balances, then you can wipe it."]),
   ("what if i can't find the phrase", ["Then don't wipe the old phone. Create a new wallet, write its phrase on paper, and move your funds over while the old one still works. Without the phrase, a wiped phone means the funds are gone."]),
   ("will the old phone still work", ["Yes, until you remove the wallet or wipe it. Both devices would hold the same key, so only keep it on devices you trust."])])

E("lost_no_phrase", 5, [r"gone|can'?t recover|cannot recover|nobody can recover|no one can recover", r"phrase"],
  ["i lost my phone and i never wrote down my seed phrase", "i didn't back up my phrase, can hermit get my funds back", "forgot to save the {words} words and my phone broke",
   "can hermit support recover my wallet without the seed", "is there any way to recover without the phrase", "lost seed phrase and phone",
   "what if i never saved my seed phrase and my phone dies", "my phone is dead and i have no backup of the recovery phrase"],
  ["I'm sorry, but without the phrase the funds are gone. Hermit can't recover the key, and nobody else can either, since it only ever lived on your device. If the phone still works, move your funds to a new wallet now and write that phrase on paper.",
   "Unfortunately, no. The key isn't in iCloud or on any server, so without your written phrase the wallet can't be restored and the funds are gone. If the device still turns on, get your funds out to a new wallet and write the new phrase down.",
   "That's the hard part of self-custody: if both the phone and the phrase are lost, nobody can recover the funds, including Hermit. If the phone still works, move everything to a new wallet today and keep the new phrase on paper, offline."],
  [("is there really nothing", ["I'm afraid not. The key only existed on your device, and Hermit never had a copy. If the phone still turns on, move your funds out right away."]),
   ("what if the phone still turns on", ["Then you still have a chance. Create a new wallet, write its phrase on paper, and send your funds to it before anything goes wrong with the phone."]),
   ("can a data recovery shop get it", ["I wouldn't count on it. The key is sealed by the Secure Enclave and meant to be unreadable, so shops generally can't pull it out. Your written phrase is the only reliable way."])])

# ---- (6) impermanent loss -------------------------------------------------------------------------------
E("il_what", 6, [r"pool|amm", r"\bhold|\bheld", r"withdraw|realis|realiz"],
  ["what is impermanent loss", "explain impermanent loss", "what does impermanent loss mean in defi", "impermanent loss eli5", "whats IL in liquidity pools",
   "how does impermanent loss work", "what is impermanent loss in uniswap", "imperment loss meaning", "can you explain impermanent loss simply",
   "what's the deal with impermanent loss", "impermanent loss, what is it", "i keep hearing about impermanent loss", "how do liquidity providers lose money",
   "im thinking about providing liquidity, what's impermanent loss", "what is impermanent loss and how do i avoid it", "define impermanent loss", "whats impermanent loss in an amm pool"],
  ["Impermanent loss happens when you put two tokens into an AMM liquidity pool, like Uniswap, and their prices move apart. The pool ends up leaving you with more of the weaker token, so you have less value than if you'd just held. It's only realised when you withdraw, and trading fees can offset it.",
   "In an AMM pool such as Uniswap you deposit two tokens, and the pool keeps rebalancing them as the price moves. If one token pulls away from the other, you end up with more of the weaker one than holding would have given you. The loss is only locked in when you withdraw, and fees you earn can make up for it.",
   "Impermanent loss is what a liquidity provider loses compared with simply holding when the two tokens in an AMM pool change price against each other. Say one token {kverb} against the other in an evenly split pool: {expr} gives {r4}, so you'd be about {pct}% behind holding. It's only realised if you withdraw, and fees can offset some of it.",
   "Impermanent loss is the gap between what you'd have by providing liquidity to an AMM pool and what you'd have by just holding the same two tokens. It appears when their relative price changes, and it's called impermanent because it's only realised when you withdraw. Trading fees can offset it, but not always.",
   "A liquidity pool holds two tokens and rebalances as the market moves. When one token's price changes a lot against the other, your share of the pool ends up with more of the weaker token than you'd have by holding. It's a paper loss until you withdraw, and the fees you earn can offset it.",
   "Picture an AMM pool of two tokens, split evenly. If the price of one {kverb} against the other, you'd end up about {pct}% worse off than if you'd just held, from {expr} = {r4}. That's impermanent loss, realised only when you withdraw, and fees can offset it.",
   "It's the cost of providing liquidity to an AMM pool of two tokens. If their prices move apart, the pool leaves you holding more of the weaker one, so you're behind holding. The loss only becomes real when you withdraw, and trading fees might cover it."],
  [("so is it permanent?", ["Not until you withdraw. If the two prices in the pool move back to where they started, the loss shrinks to nothing. If you withdraw while they're apart, it becomes a real loss compared with holding."]),
   ("how do i avoid it", ["You can't avoid it completely if you provide liquidity to a pool of two tokens that move apart. Pairs of tokens that stay close in price, like two stablecoins, have much less of it. Or skip pools and just hold, or lend."]),
   ("is it the same as lending my coins out", ["No. Lending is different: you deposit one token and earn interest on it, with no pool of two tokens being rebalanced. Impermanent loss is specific to AMM liquidity pools."])],
  slots=s_il, bad=[r"\baave\b"], w=1.7)

E("il_lp", 6, [r"pool", r"impermanent|hold", r"fee"],
  ["should I LP {pair}", "is providing liquidity to {pair} worth it", "thinking of lp'ing in a {pair} pool, good idea?", "should i put my tokens in a uniswap pool",
   "is liquidity providing profitable", "is it worth being a liquidity provider", "should i provide liquidity with my eth", "lp or just hold?", "is lp'ing safe",
   "do liquidity pools make money", "what are the risks of providing liquidity", "is {pair} lp worth it", "is farming lp pairs a good idea",
   "should i LP or just hodl", "any downside to putting eth in an lp"],
  ["It can earn fees, but you take on impermanent loss: if one token in the pool moves a lot against the other, you end up with less than holding would have given you. It only pays off when the fees you earn beat that loss, which depends on trading volume and how far the price moves. Only use money you can accept losing, and remember smart-contract risk too.",
   "It's a trade-off. A pool pays you a share of trading fees, but if the two tokens drift apart in price you end up with more of the weaker one, and that impermanent loss can be bigger than the fees. A pair like {pair} can earn decent fees, but a big move hurts, so size it carefully.",
   "That depends on how much the price moves. You earn fees from the pool, but impermanent loss eats into that if one token runs away from the other, and holding would have done better. Check the pool's real fee history and keep to what you can afford to lose.",
   "Providing liquidity isn't free money. The pool's fees can look attractive, but when one token's price changes against the other you hold more of the weaker one, so you can end up behind just holding. Compare the fees to that risk before you go in.",
   "Providing liquidity to a pool earns fees but carries impermanent loss. For scale, if one token {kverb} against the other in an evenly split pool, {expr} gives {r4}, so you'd be about {pct}% behind just holding. The pool's fees have to beat that for it to pay off."],
  [("what about a stablecoin pair", ["Pools of two stablecoins usually have much less impermanent loss, because their prices stay close together. You still carry smart-contract risk and depeg risk if one of them loses its peg. Fees are usually lower too."]),
   ("how do i know if the fees are enough", ["Compare the fees the pool pays to the impermanent loss you'd suffer if the price moved. High trading volume helps, big price swings hurt. A high advertised APR alone doesn't tell you."]),
   ("what's the other risk", ["Smart-contract risk: a bug in the pool's code could lose funds. There's also a risk that you can't withdraw at a good moment. That's on top of impermanent loss."])],
  slots=s_il, bad=[r"\baave\b"], w=1.4)

E("il_realised", 6, [r"withdraw", r"pool|amm|liquidity"],
  ["is impermanent loss permanent", "do i lose money if i dont withdraw from the pool", "what if the price comes back, does the loss go away", "why is it called impermanent",
   "is impermanent loss real", "can impermanent loss disappear", "if i stay in the pool is the loss locked in", "does il only count when you withdraw",
   "price went back to where i entered the pool, is my impermanent loss gone"],
  ["It's only realised when you withdraw. If the prices of the two tokens in the pool move back to where they started, the loss shrinks to nothing. If you withdraw while they're apart, it becomes a real loss versus holding.",
   "That's why it's called impermanent: while you stay in the pool, it's a paper loss that changes with the price. If the pair returns to its original ratio, it's gone. Withdraw while they're apart, and you lock it in.",
   "It only becomes real when you withdraw. As long as you stay in the liquidity pool, the gap versus holding moves up and down with the price, and it can shrink back to zero. Fees you earn in the meantime can offset it as well.",
   "It's only realised on withdrawal. In an evenly split pool where one token {kverb}, the gap versus holding is about {pct}% ({expr} gives {r4}), but it stays on paper while you remain in the pool. If the price ratio returns, the gap closes."],
  [("so i should just wait?", ["Waiting can help if the prices come back, but they might not. Meanwhile, the pool keeps earning fees, which can offset the loss. There's no guarantee either way."]),
   ("do fees count?", ["Yes. The trading fees you earn in the pool add to your position and can offset impermanent loss or beat it. The more volume, the more fees."]),
   ("what if one token goes to zero", ["Then the pool leaves you holding almost all of the failing token, and your loss compared with holding can be huge. That's the worst case of providing liquidity to a pool of two tokens."])],
  slots=s_il, bad=[r"\baave\b"])

E("il_fees", 6, [r"fee", r"offset|cover|beat|make up|make-up", r"pool|amm|liquidity"],
  ["do fees cover impermanent loss", "do lp fees make up for impermanent loss", "are pool fees worth the impermanent loss", "how do liquidity providers earn",
   "can fees offset il", "when does lp profit beat il", "is the apr on a uniswap pool worth it considering impermanent loss", "how much do lps earn in fees"],
  ["Sometimes. Pools pay liquidity providers a share of the trading fees, and with enough volume those fees can offset impermanent loss or beat it. If the price moves a lot and volume is low, the loss can win. A high advertised APR on its own doesn't tell you which.",
   "It depends on the pool. A liquidity pool pays its providers part of every trade's fee, which can cover impermanent loss when volume is high. When the price swings hard and trading is quiet, the loss can be bigger than the fees.",
   "They can, but they don't have to. You earn fees from the pool's trading, and the fees can offset the loss from the two tokens moving apart, or fall short of it. Look at real volume, not just the headline yield."],
  [("how do i check that", ["Look at the pool's trading volume and fees over time, and compare them with how much the two prices have moved since you'd have entered. A high advertised yield can disappear when the price swings."]),
   ("is a stable pair better", ["A pair of stablecoins has much less impermanent loss, but it pays lower fees too. You still have smart-contract and depeg risk. It's a trade-off."]),
   ("what's a good fee level", ["There's no fixed number. It depends on volume and how far the pair moves. If fees keep coming in faster than the price gap grows, the pool pays off."])],
  bad=[r"\baave\b"])

E("il_example", 6, [r"pool|amm", r"\bhold|\bheld", r"withdraw|realis|realiz|fee"],
  ["how much do i lose if one token {kverb} in a pool", "give me an example of impermanent loss", "show me the math for impermanent loss", "impermanent loss example with numbers",
   "if one token {kverb} in a liquidity pool how bad is impermanent loss", "what's the impermanent loss formula", "how much is il if the price goes {kx} in {pair}",
   "if eth {kverb} against usdc in a pool, what's my impermanent loss", "can you calculate impermanent loss for a {kx} move", "worked example of impermanent loss please",
   "how big is impermanent loss when one side {kverb}"],
  ["Take an evenly split AMM pool. If one token {kverb} against the other, the loss versus holding is {expr}, which is {r4}, about {pct}%. That's only realised when you withdraw, and trading fees can offset some of it.",
   "For an evenly split pool, the formula is {expr}. With a {kx} move it comes to {r4}, so you'd be about {pct}% behind just holding. The loss only becomes real when you withdraw from the pool, and fees can cover part of it.",
   "Say the price of one token {kverb} against the other in a two-token pool split evenly. The maths, {expr}, gives {r4}, which means about {pct}% less than holding. You only realise it on withdrawal, and the fees the pool pays you can soften it.",
   "With an AMM pool split evenly, a {kx} price move gives {expr} = {r4}. In plain terms you'd have roughly {pct}% less than if you'd held the two tokens. It's not locked in until you withdraw, and fees earned can offset it."],
  [("what if it goes even higher", ["The loss grows, but slowly at first and then faster, because the pool keeps selling the token that's rising. The bigger the price gap, the bigger the shortfall against holding. Fees can still offset some of it."]),
   ("does it work the same going down", ["Yes. It depends on how far the two prices move apart, not which direction. A move down by the same ratio gives the same percentage shortfall against holding."]),
   ("and if fees were big enough", ["Then the fees can cover the loss or beat it, and you come out ahead of holding. That depends on how much volume the pool sees while you're in it."])],
  slots=s_il, bad=[r"\baave\b"], w=1.2)

# ---- (7) Aave ------------------------------------------------------------------------------------------------
E("aave_what", 7, [r"lend", r"deposit|supply"],
  ["what is aave", "how does aave work", "what's aave in defi", "is aave a bank", "explain aave", "what do I get from depositing in aave", "what is aave v3",
   "how do you earn on aave", "aave what is it exactly", "is aave a dex"],
  ["Aave is a lending protocol. You deposit a token such as USDC, borrowers pay interest to use it, and you earn a share as a variable rate. It's lending, not a liquidity pool of two tokens, so the risks are smart-contract bugs, market moves and depegs.",
   "It's a lending market, not a bank. You supply a token, other users borrow against collateral, and you earn interest that moves with demand. Nobody guarantees the rate, and smart-contract risk is part of the deal.",
   "Aave lets you lend crypto and earn interest, or borrow against collateral. When you deposit one token, borrowers pay for using it and the rate floats with supply and demand. It carries smart-contract, market and depeg risk, and it isn't a DEX."],
  [("is it safe", ["Safer than many options, but not risk free. Smart-contract bugs, market moves and a stablecoin losing its peg can all hurt, and the rate is variable. Only deposit what you can afford to lose."]),
   ("which chains can i use it on in hermit", ["Hermit supports Aave v3 deposits on Ethereum, Base and Arbitrum. You can withdraw part of your deposit or all of it. Each move needs your approval with {BIO}."]),
   ("do i get a fixed rate", ["No, the rate is variable. It changes with how much of the pool is borrowed, so it can go up or down. Look at the current rate before you deposit, and expect it to move."])])

E("aave_il", 7, [r"\bno\b|not", r"lending|lend", r"amm|pool|two tokens"],
  ["does aave have impermanent loss", "is there impermanent loss when i deposit usdc in aave", "do i get il on aave", "aave impermanent loss?",
   "if i supply eth to aave will i suffer impermanent loss", "aave vs uniswap lp, which has impermanent loss", "does lending on aave expose me to il", "is aave an lp?",
   "aave pool impermanent loss risk?", "i deposited in the aave pool, is that liquidity providing?", "is depositing into aave the same as providing liquidity",
   "does aave leave me with less than holding like a uniswap pool", "do aave deposits suffer impermanent loss?", "aave lending, any impermanent loss there?"],
  ["No. That kind of loss comes from AMM pools where you hold two tokens that get rebalanced as prices move. Aave is a lending market, so you deposit one token and earn interest on it. Your real risks are smart-contract bugs, market moves and a stablecoin losing its peg, and the rate is variable.",
   "No impermanent loss applies to plain lending, because there's no pool of two tokens being rebalanced against each other. A deposit on Aave is a loan: it's one token earning interest, not a liquidity position. The risks to watch are smart-contract, market and depeg risk, and the variable rate.",
   "No, it doesn't. Impermanent loss belongs to AMM pools where you supply two tokens and the pool shifts the mix as prices move. Aave is lending, so your deposit stays the one token you put in, plus interest at a variable rate. The risks are smart-contract, market and depeg.",
   "No, and the reason is the design: it's a lending market, not an AMM. Impermanent loss only comes from AMM liquidity pools where the price of two tokens moves apart. With Aave the risks are different: smart-contract bugs, market moves, depegs and a variable APY."],
  [("so what are the risks there", ["Smart-contract bugs, market risk if the token you deposit falls in price, and depeg risk if a stablecoin loses its dollar value. The rate you earn is variable too. None of it is guaranteed."]),
   ("what about borrowing", ["Borrowing adds liquidation risk: if your collateral falls too far in value, it can be sold to repay the loan. Keep a safe margin and watch it. Plain lending doesn't have that risk."]),
   ("which one has more risk, lp or lending", ["They're different risks, so it depends. An LP pool adds impermanent loss on top of smart-contract risk, while plain lending has market, depeg and smart-contract risk with variable rates. Neither is risk free."])],
  w=1.7)

E("aave_risks", 7, [r"smart.contract", r"depeg|peg", r"variable"],
  ["what are the risks of aave", "is aave safe", "can i lose money lending on aave", "is aave apy fixed", "what happens if usdc depegs on aave", "is aave risk free",
   "how risky is it to deposit stablecoins in aave", "what could go wrong if i lend on aave", "why does the aave rate keep changing"],
  ["Aave isn't risk free. The main risks are smart-contract bugs, market risk if the token you deposit falls in price, and depeg risk if a stablecoin loses its dollar value. The rate you earn is variable and changes with demand.",
   "It can go wrong in a few ways: a bug in the smart contracts, the price of what you deposited dropping, or a stablecoin losing its peg. The APY isn't fixed, it's variable, so expect it to move. Only deposit what you can accept losing.",
   "Not risk free. Think smart-contract risk, market risk and depeg risk, plus a variable APY that moves with how much is borrowed. A stable-looking rate today can change tomorrow."],
  [("is usdc safer than eth there", ["USDC avoids the price swings of ETH, but it carries depeg risk if it ever loses its dollar value. Both carry smart-contract risk. ETH adds market risk on top."]),
   ("what happens if it depegs", ["If a stablecoin you deposited loses its peg, your deposit is worth less than you put in, even if the number of tokens stays the same. Aave can't change that. It's one of the reasons it isn't risk free."]),
   ("can i take it out anytime", ["Usually, yes, you can withdraw part or all of your deposit when you want. Withdrawals depend on there being enough in the pool, which is another risk to know about. Each one needs your approval with {BIO}."])])

# ---- (8) swaps and bridges --------------------------------------------------------------------------------
E("swap_route", 8, [r"near intents"],
  ["how do swaps work in hermit", "how does hermit bridge", "what do you use for swaps", "what powers the swaps", "how does the swap route work", "what is near intents",
   "do you use 1inch or uniswap for swaps", "which dex does hermit use", "how does bridging work in the app", "what's the swap provider", "where do the swaps come from",
   "who fills my swap", "does hermit swap through uniswap?", "what route does hermit use to bridge", "how do cross chain swaps work here"],
  ["Swaps and bridges in Hermit go through NEAR Intents. You ask for what you want, a quote shows the amount and the cost, and nothing moves until you approve it with {BIO}. A quote on its own doesn't commit you to anything.",
   "Hermit uses NEAR Intents for swaps and bridges. I can get you a quote first, and when you want to go ahead I prepare it and open a review screen for you to approve with {BIO}.",
   "They run on NEAR Intents, which also handles moving tokens between chains. I only prepare the swap, and you approve it with {BIO} on the review screen. Nothing happens before that.",
   "NEAR Intents is the route for swaps and bridges in Hermit. It finds you a quote across chains, and you see the amount and cost before you approve anything with {BIO}."],
  [("is it safe", ["Every swap or bridge needs your own approval with {BIO}, and you can see the quote and cost first. That doesn't remove all risk, so check the amounts and the chain before you approve."]),
   ("can i just get a quote", ["Yes. Ask for a quote and I'll fetch it without preparing anything. It shows the amount you'd get and the cost, and nothing moves."]),
   ("can i bridge between chains", ["Yes, bridges between supported chains also use NEAR Intents. You get a quote, then I prepare it and you approve it with {BIO}."])],
  w=1.3)

E("swap_provider", 8, [r"near intents"],
  ["is it relay", "do you use relay for bridging", "is hermit swapping through relay or something else?", "which bridge is this, relay?", "is this lifi",
   "what bridge provider is used", "does the swap use 0x", "is the route relay or something else", "is it across or relay under the hood", "which aggregator are you using for swaps"],
  ["Swaps and bridges in Hermit use NEAR Intents. You see the quote and cost first, and nothing moves until you approve with {BIO}.",
   "It's NEAR Intents. That's the route for swaps and bridges in the app, and each one needs your approval with {BIO} before it happens.",
   "Hermit uses NEAR Intents for swaps and bridges. I only prepare them, and you approve with {BIO}."],
  [("how do i see the route", ["When I fetch a quote or prepare a swap, the result names the route, and I'll tell you what it says. It's NEAR Intents for swaps and bridges."]),
   ("are the fees different", ["The quote shows the all-in cost before you approve, so you can compare it against what you expect. I can fetch a quote for any swap you're considering."]),
   ("can i choose another route", ["Hermit uses NEAR Intents for swaps and bridges, and I don't have a way to switch the route. You can ask for a quote and decide whether to approve it."])])

# ---- (9) approvals and limits -----------------------------------------------------------------------------------
E("approve_all", 9, [r"approv", r"prepare|review", r"face id|touch id"],
  ["do i need to approve every transaction", "can the ai send money without my approval", "will hermit ever move my funds on its own", "can the ai spend my money without asking",
   "does every trade need {BIO}", "what happens when i ask the ai to send crypto", "do i have to confirm swaps", "can someone make the assistant send my money",
   "how does approval work", "is there a confirm step before sending", "does the ai press the buttons for me", "can the ai make a trade while i'm not looking"],
  ["Every move needs your approval. The AI only prepares it and opens a review screen, and nothing happens until you approve with {BIO}. Each action needs its own approval.",
   "Yes. I can prepare a send, swap or trade, but it only happens when you review it and approve with {BIO}. There's no way for me to approve for you.",
   "No, it can't act on its own. I prepare the move, a review screen opens, and you approve it with {BIO}, one action at a time. Until you do, nothing is sent.",
   "Nothing moves without you. The assistant prepares and you approve each action with {BIO} on the review screen."],
  [("what if i don't approve", ["Then nothing happens. The review stays open or you can cancel it, and no funds move. You're always in control of the last step."]),
   ("what does the review screen show", ["It shows the details of the action I prepared, such as the amount and where it goes, so you can check before you approve. If anything looks wrong, cancel it."]),
   ("can i approve many at once", ["Each action needs its own approval, so you approve them one by one. A batch send is a single action with one approval, but you still review it first."])],
  w=1.3)

E("ai_limits", 9, [r"daily", r"cap|limit", r"face id|touch id|approv"],
  ["is there a limit on what the ai can spend", "can the assistant drain my wallet", "what stops the ai from sending everything", "any spending limits?",
   "what's the daily limit", "how much can the ai move per day", "is there a cap on what the assistant can send", "could a bug make the ai send all my funds"],
  ["Yes. The app has a daily spend limit and a cap per move for the assistant, and every move still needs your approval with {BIO}. I don't have the exact amounts to quote.",
   "There are two guards: a daily spend limit and a per-move cap for the assistant. On top of that, nothing moves until you approve with {BIO}. I don't know the exact numbers.",
   "The assistant has a daily spend limit and a cap on each move, and you approve every action with {BIO}. So it can't empty your wallet on its own. I can't tell you the exact limits from here."],
  [("can i change the limit", ["I don't know of a way to change it from here, so I won't guess. What I can say is that each move still needs your approval with {BIO}."]),
   ("what happens when i hit it", ["The assistant can't prepare more moves past the limit. I'll tell you if something gets blocked. You're still the one who approves every action."]),
   ("so i'm fully protected?", ["The limits and your approval help a lot, but they don't cover scams or a wrong address you approve yourself. Check each review before you approve with {BIO}, because transactions can't be reversed."])])

# ---- (10) on device -----------------------------------------------------------------------------------------------
E("device_private", 10, [r"on (your|this|the) (device|iphone|mac|phone)|runs (entirely )?on", r"nothing|stays|doesn'?t leave|never leaves|isn'?t sent"],
  ["does hermit send my chats to a server", "is the ai on device", "does anything i type leave my phone", "is my conversation private", "does the ai run locally",
   "are my messages sent to openai", "does hermit use chatgpt", "is it cloud ai or local", "where does the ai run", "who can read my chats",
   "is my chat history sent anywhere", "do you log what i ask you"],
  ["The AI runs on your device, and nothing you type leaves it. There's no cloud model behind this chat.",
   "It runs entirely on this device. What you type stays here, and it isn't sent to a server or a cloud model.",
   "Local. The model runs on your device, so your messages stay on it and nothing typed goes to a cloud service.",
   "Everything stays on your device. The AI runs here, and what you type to it doesn't leave."],
  [("so it works without internet?", ["The AI itself runs on your device, so chatting doesn't need internet. Anything that needs live data, like prices, or that sends a transaction does need a connection."]),
   ("what about my wallet data", ["Your key stays sealed on your device too, in the Secure Enclave, and neither Hermit nor the AI ever sees it. The app only uses the network for things like live prices and moving funds."]),
   ("can anyone read my chats", ["Nothing you type leaves the device, so nobody else can read it from a server. Anyone who can unlock your phone could see it, so keep your device locked."])],
  w=1.2)

E("device_offline", 10, [r"on (your|this|the) (device|iphone|mac|phone)", r"internet|connection|offline|network"],
  ["does hermit work offline", "can i chat with the ai in airplane mode", "do i need wifi to use the ai", "does the ai need internet", "will it work without signal",
   "does the chat work in flight mode", "can i use the ai on a plane"],
  ["The AI runs on your device, so chatting doesn't need internet. Anything that needs live data, like prices, or that sends a transaction does need a connection.",
   "Chat works offline because the model lives on your device. Live prices, balances and sending anything need a connection.",
   "Yes for chat: the AI runs on this device, so it works without signal. The parts that fetch live data or send a transaction need the internet."],
  [("so prices won't load?", ["Right. Live prices need a connection, so without one I can't give you a current price. I won't guess a number."]),
   ("does it use my data plan?", ["Chatting itself doesn't, since the AI runs on your device. Fetching live prices or sending a transaction uses a small amount of data."]),
   ("is it slower offline", ["Chat speed depends on your device, not your connection, since the model runs on it. Older devices can be slower. Online tasks depend on your connection."])])

# ---- (11) ENS -----------------------------------------------------------------------------------------------------
E("ens_send", 11, [r"\.eth|ens", r"yes|works|can|support"],
  ["can i send to an ens name", "does hermit support .eth names", "can i send to vitalik.eth", "send to alice.eth works?", "do ens domains work for sending",
   "can you resolve ens names", "can i pay a .eth address", "is ens supported", "can i type a .eth name instead of the 0x address", "does hermit understand ens names",
   "do .eth names work when sending crypto here"],
  ["Yes, .eth names work as send targets. Type the name in your message and Hermit will prepare the send to it. Check that the name is spelled exactly right before you approve, because one wrong letter is a different address.",
   "Yes, ENS names are supported for sends. Give me the .eth name as you'd type it and I'll prepare the send, which you approve with {BIO}. Double-check the spelling first, since a similar-looking name goes to someone else.",
   "It does. A .eth name works in place of a 0x address when you send. Check every letter, and approve the review with {BIO}."],
  [("do i need the full address too", ["No, the .eth name is enough. I pass it on exactly as you typed it, so type it carefully. You'll still review and approve the send with {BIO}."]),
   ("what if the name is wrong", ["A name that doesn't exist won't work, and one with a typo may point to someone else, so check each letter. Transactions can't be reversed once approved."]),
   ("what if i mistype it", ["A typo can point to a different address or to nothing, so check every letter before you approve. Transactions can't be reversed once approved. Copy the name from where you got it if you can."])])

# ---- (12) Aave chains and withdraw ----------------------------------------------------------------------------------
E("aave_chains", 12, [r"ethereum", r"base", r"arbitrum"],
  ["which chains can i deposit to aave on", "can i use aave on base", "does hermit support aave on polygon", "what networks does aave work on in hermit", "aave deposits on arbitrum?",
   "is aave on ethereum mainnet supported in the app", "which networks for aave in hermit", "can i deposit in aave on solana", "aave on bnb chain in hermit?"],
  ["Aave v3 deposits in Hermit are on Ethereum, Base and Arbitrum. Each deposit is its own action, and you approve it with {BIO}.",
   "Hermit supports Aave v3 deposits on Ethereum, Base and Arbitrum. You can withdraw part or all of a deposit, and each move needs your approval with {BIO}.",
   "You can deposit to Aave v3 on Ethereum, Base and Arbitrum from Hermit. I prepare it and a review opens for you to approve with {BIO}."],
  [("can i withdraw part of it", ["Yes, a withdrawal can be partial or the full amount. Tell me how much, or say all of it, and I'll prepare it for your approval."]),
   ("what's the rate", ["The rate is variable and moves with demand, so I can't give you a fixed number here. When I prepare a deposit, the result includes the current rate and I'll tell you what it says."]),
   ("which chain is cheapest", ["I can't compare fees without checking, and fees change. Base and Arbitrum are usually cheaper than Ethereum, but check the review before you approve."])])

E("aave_withdraw", 12, [r"part|partial", r"all|full|everything", r"approv|review"],
  ["can i withdraw part of my aave deposit", "can i take everything out of aave at once", "is partial withdrawal from aave possible", "do i have to withdraw all of it from aave",
   "can i take just some of my usdc out of aave", "how much can i pull from aave at a time", "are aave withdrawals all or nothing"],
  ["Yes, you can withdraw part of your deposit or all of it. Tell me an amount for a partial withdrawal, or say you want everything out. It opens a review, and you approve it with {BIO}.",
   "Both work. A withdrawal can be partial or the full balance, and each one is prepared for your review and approved with {BIO}.",
   "It's your choice: part of the deposit or all of it. Give me the amount, or say all, and I'll prepare the withdrawal for you to approve."],
  [("how long does it take", ["I don't have a fixed time. It depends on the network, and you'll see it in Activity in Wallets once you've approved it. If the pool is short on funds, the withdrawal may be limited."]),
   ("is there a fee", ["There's a network fee for the transaction, like any move on a chain. The review screen shows the details before you approve with {BIO}."]),
   ("what if i only say withdraw my usdc", ["Then I'll treat it as everything you have in Aave in USDC and prepare the full withdrawal. If you want less, tell me an amount and I'll use that."])])

# ---- (13) chains --------------------------------------------------------------------------------------------------------
E("chains_list", 13, [r"ethereum", r"base", r"arbitrum", r"bnb", r"polygon", r"monad", r"solana", r"near"],
  ["which chains does hermit support", "what networks can i use", "does hermit work with solana", "is polygon supported", "does hermit support base and arbitrum",
   "can i use hermit on bnb chain", "is monad supported", "does it support near", "supported blockchains?", "what chains?", "what networks are supported in hermit",
   "does hermit do ethereum mainnet", "can i hold sol in hermit", "does hermit work on arbitrum"],
  ["Hermit supports Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR. Swaps and bridges between them go through NEAR Intents.",
   "These chains: Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR. Each has its own balance, and you approve every move with {BIO}.",
   "You can use Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR in Hermit. Ask me for your balances and I'll show where your tokens are.",
   "Hermit works on Solana, NEAR, Ethereum, Base, Arbitrum, BNB Chain, Polygon and Monad. Moving tokens between them is a bridge through NEAR Intents."],
  [("what about bitcoin", ["Bitcoin isn't on the list of chains Hermit supports. The chains are Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR."]),
   ("can i move between them", ["Yes. Swaps and bridges between supported chains go through NEAR Intents. I can fetch a quote first, and you approve any move with {BIO}."]),
   ("do i need a different wallet for each", ["No. One Hermit wallet covers the supported chains, and the balances are listed per chain. Your key stays on your device, protected by the Secure Enclave."])])

# ---- (14) trading terms -------------------------------------------------------------------------------------------------
E("perp_what", 14, [r"perp|perpetual", r"long|short|leverage|funding"],
  ["what is a perp", "what are perpetual futures", "perp vs spot", "what does perp mean in crypto", "explain perps", "what's a perpetual contract",
   "how is a perp different from buying the coin", "whats perp trading", "perps explained simply", "what is a perp dex"],
  ["A perp, or perpetual future, is a contract that tracks a coin's price without you owning the coin, and it has no expiry date. You can go long or short, usually with leverage. A funding payment between longs and shorts keeps its price close to the real market.",
   "It's a way to trade a coin's price without holding the coin. A perp never expires, you can bet up or down, and leverage can make your position bigger than your money. Funding payments tie it to the spot price.",
   "A perpetual contract follows the price of an asset, like a futures contract that never ends. You can go long or short, and you can use leverage, which also increases the risk of liquidation. Funding keeps it near the spot price."],
  [("is it risky", ["Yes, especially with leverage. Losses scale with the full position size, and if the price moves against you far enough, you can be liquidated and lose your margin. Size it small and set a stop."]),
   ("how is it different from spot", ["With spot you buy the actual coin and own it. With a perp you hold a contract on its price, which lets you short and use leverage but adds funding costs and liquidation risk."]),
   ("where can i trade them", ["Hyperliquid is where Hermit prepares perp trades. I prepare the order, and you approve it with {BIO} on the review screen."])])

E("leverage", 14, [r"leverage", r"loss|lose|liquidat|risk"],
  ["what is leverage", "how does {lev}x leverage work", "what does {lev}x leverage mean", "explain leverage like im new", "is leverage dangerous", "what happens at {lev}x with ${margin}",
   "leverage trading explained", "how much can i lose with leverage", "what does it mean to trade with ${margin} at {lev}x", "what is {lev}x on a perp"],
  ["Leverage lets you open a bigger position than your own money. With ${margin} at {lev}x you control ${pos}. Gains and losses scale with the full position, so a move of about {move}% against you can wipe out your margin.",
   "Leverage is borrowing to make a position bigger. At {lev}x, ${margin} of your own money controls ${pos}, so every price move counts {lev} times as much. A move of roughly {move}% the wrong way can liquidate you.",
   "Leverage multiplies both profit and loss. Put ${margin} in at {lev}x and you hold ${pos} of exposure, which means about {move}% against you can erase your margin. The higher the leverage, the less room you have."],
  [("what's a safe amount", ["There's no safe number, but lower leverage gives your trade more room before liquidation. Many people keep it low and size the position so a stop-loss costs only a small part of their account."]),
   ("what if i use less", ["Lower leverage means a smaller position for the same margin, so you earn and lose less per price move, and the liquidation price is farther away. It's the simplest way to cut risk."]),
   ("can i lose more than i put in", ["With leveraged perps you normally lose the margin behind the position, not more, because it's closed at liquidation. In very fast markets the price can jump past the liquidation point, so treat it as a risk. Keep position sizes small."])],
  slots=s_lev, w=1.2)

E("funding", 14, [r"funding", r"long", r"short"],
  ["what is funding rate in perps", "what does funding mean on hyperliquid", "why am i paying funding", "positive funding who pays", "explain funding rate",
   "what is funding", "do i pay funding on a long", "how does funding work on a ${pos} position at {rate}%", "what does a funding rate of {rate}% mean"],
  ["Funding is a periodic payment between longs and shorts that keeps a perp's price close to the spot price. When the rate is positive, longs pay shorts, and when it's negative, shorts pay longs. For example, on a ${pos} position, a rate of {rate}% costs about ${pay} each funding period.",
   "It's the fee longs and shorts pay each other to keep the perp near the spot price. A positive rate means longs pay shorts, a negative one the other way round. For example, at {rate}% on ${pos}, that's about ${pay} per funding period.",
   "Think of funding as a balancing payment. If the rate is positive, longs pay shorts, and if it's negative, shorts pay longs. For example, with ${pos} open and a rate of {rate}%, you'd pay or receive about ${pay} each period."],
  [("who gets the money", ["The other side of the trade gets it, not the exchange. When funding is positive and you're long, shorts receive your payment. If you're short, you'd be the one receiving it."]),
   ("can i check the rate", ["Yes, I can look up the current funding on a market for you. Tell me which coin and I'll fetch it."]),
   ("is it a big cost", ["Usually small per period, but it adds up if you hold a leveraged position for a long time. A high rate is worth knowing before you open a trade."])],
  slots=s_fund)

E("liquidation", 14, [r"liquidat", r"margin|position"],
  ["what is liquidation", "what does getting liquidated mean", "how does liquidation work", "when do i get liquidated at {lev}x", "what is a liquidation price",
   "what happens when you get liquidated", "how close is liquidation at {lev}x if i enter at ${entry}", "explain liquidation in perps"],
  ["Liquidation is when the exchange closes your leveraged position because your margin can no longer cover the losses. At {lev}x, a long opened at ${entry} is liquidated roughly when the price falls to about ${liq}, a bit sooner in practice because of fees. Lower leverage puts the liquidation price further away.",
   "When your losses eat the margin behind a leveraged trade, the position is closed for you, and that's liquidation. For a long at ${entry} with {lev}x, the rough liquidation price is ${liq}, and in practice it's a little higher. Less leverage means more room.",
   "It's a forced close of your position when your margin is nearly used up. At {lev}x, a long entered at ${entry} gets liquidated around ${liq}, slightly earlier once fees and margin rules are counted. You lose the margin in that trade."],
  [("how do i avoid it", ["Use lower leverage, keep extra margin in the account, and set a stop-loss well before the liquidation price. Smaller positions give you more room."]),
   ("what about shorts", ["Same idea, but the price rising is what hurts. A short is liquidated when the price rises far enough that the margin can't cover the loss."]),
   ("can i see my liquidation price", ["Yes, if you have an open position on Hyperliquid I can look it up for you and tell you how far away it is. Just ask me to check."])],
  slots=s_liq, w=1.2)

E("slippage", 14, [r"slippage", r"price"],
  ["what is slippage", "what does slippage mean in a swap", "why did i get less than quoted", "what's slippage tolerance", "how does slippage work", "what causes slippage",
   "is high slippage bad", "what happens with {slip}% slippage on a ${amt} trade", "what is {slip}% slippage"],
  ["Slippage is the gap between the price you expect and the price you actually get, which happens when the market moves or your order is big compared with the liquidity. With {slip}% slippage on ${amt}, you could receive about ${loss} less than expected. A tight limit protects you, but a trade can fail if the price moves past it.",
   "Slippage is the difference between the quoted price and what you end up with. At {slip}% on ${amt}, that's up to about ${loss}. Setting it lower protects you but makes a failed trade more likely.",
   "When you trade, the price can move before it fills, and slippage is how much it moved against you. With {slip}% on ${amt} you could lose about ${loss} to it. Bigger orders in thin markets slip more."],
  [("how do i keep it low", ["Trade smaller sizes, use liquid markets, and avoid moments of big volatility. A limit order controls the price completely, but it may not fill."]),
   ("is it the same as fees", ["No. Fees are a charge for the service. Slippage is the price changing between the quote and the fill, and it can go for or against you."]),
   ("do you show it when i swap", ["When I get a quote, it shows the amount you'd receive and the total cost, so you can judge it before you approve. I can fetch one for any swap you're thinking of."])],
  slots=s_slip)

E("limit_order", 14, [r"limit", r"price", r"market"],
  ["what is a limit order", "limit vs market order", "difference between market and limit", "what does a limit order do", "how does a limit order work", "when should i use a limit order",
   "limit order meaning", "what does a limit buy at ${lim} do if it trades at ${now}"],
  ["A limit order sets the price you're willing to trade at, and it only fills if the market reaches that price. A market order fills right away at the best available price, which can be worse than you expected. With a coin at ${now}, a limit buy at ${lim} waits until the price falls to ${lim}.",
   "With a limit order you name the price, and it fills only at that price or better. A market order fills now at whatever the market offers. If a coin is at ${now}, a limit buy at ${lim} just sits there until the price drops to it.",
   "Limit orders give you control over price but no guarantee of a fill, while market orders give you a fill but no control over price. For example, a limit buy at ${lim} on a coin trading at ${now} only fills if it drops to ${lim}."],
  [("what if it never reaches the price", ["Then the order stays open and doesn't fill, until you cancel it or it expires. That's the trade-off: you get your price or no trade."]),
   ("which is cheaper", ["Limit orders often have lower fees and no slippage, but may not fill. Market orders fill fast and cost more in spread and slippage. It depends on whether speed or price matters more."]),
   ("can you place one for me", ["I can prepare a limit order, and you approve it with {BIO}. Tell me the coin, side, size and the price you want."])],
  slots=s_lim)

# ---- (15) revoke ----------------------------------------------------------------------------------------------------
E("revoke", 15, [r"revoke\.cash", r"no approvals (screen|page)|doesn'?t have an approvals"],
  ["how do i revoke token approvals", "revoke approvals hermit", "does hermit have a revoke approvals screen", "how to remove permissions i gave a dapp", "should i revoke approvals",
   "what are token approvals", "how do i stop a dapp spending my tokens", "is there an approvals page in hermit", "how do i check which contracts can spend my tokens",
   "i approved a sketchy site, how do i undo it"],
  ["Hermit has no approvals screen yet, so use a tool like revoke.cash to see and revoke the permissions you've given to contracts. Look up your wallet address there to see what's approved. Revoking takes a transaction, so it costs a small network fee.",
   "There's no approvals screen in Hermit yet. A tool like revoke.cash lists what your address has approved and lets you revoke it. Revoke anything you don't recognise, and expect a small network fee for each.",
   "Not in Hermit yet, since it has no approvals screen. Use a tool like revoke.cash: enter your address, review the list, and revoke the permissions you don't need. Each revoke is a transaction with a small fee."],
  [("is it urgent if i approved a scam site", ["Yes, it's worth doing soon. Revoking the approval stops that contract from spending your tokens. Also move anything valuable if you think it was a drainer, and never share your phrase."]),
   ("what are approvals again", ["When you let a dApp use a token, you approve it to spend up to some amount from your wallet. The approval stays until you revoke it, even after you leave the site. That's why old ones are worth cleaning up."]),
   ("do i need to revoke after every swap", ["Not after every one, but it's good practice to clean up old approvals, especially for sites you no longer use or don't trust. A tool like revoke.cash shows you the list."])])

# ---- (16) is hermit safe -------------------------------------------------------------------------------------------
E("hermit_safe", 16, [r"secure enclave", r"risk|can'?t promise|can'?t protect|can'?t be (undone|reversed)|no wallet"],
  ["is hermit safe", "is hermit wallet safe to use", "can i trust hermit", "is it safe to put my money in hermit", "how safe is this app", "is hermit secure",
   "what are the risks of using hermit", "should i trust an ai with my crypto", "is hermit legit", "can hermit get hacked", "is it safe to use hermit with real money",
   "how secure is hermit really"],
  ["Here's the honest picture. You hold your own key: it's sealed by the Secure Enclave on your device, every move needs your approval with {BIO}, and the AI runs on the device. The risks are yours too: lose your written phrase and the funds are gone, and scams, bad contracts and mistyped addresses can't be undone.",
   "No wallet is risk free, so here's what I can say. Your key stays on your device in the Secure Enclave, the AI never sees it, and nothing moves without your {BIO} approval. What Hermit can't protect you from is a lost phrase, a scam or a send to the wrong address, since none of those can be reversed.",
   "It's self-custody: only you hold the key, and I can't promise any app is bug-free. In Hermit the key sits in the Secure Enclave, every move needs {BIO}, and the assistant has a daily spend limit and a per-move cap. You still carry the risk of phishing, smart-contract bugs and losing your phrase.",
   "I can't promise any app is perfectly safe. What I can tell you: the key is stored on your device and sealed by the Secure Enclave, the AI can't see it, and every move needs your approval. Keep your phrase offline and double-check every address and site, because crypto transactions can't be reversed."],
  [("so what's the biggest risk", ["Mostly you: losing your written phrase, falling for a scam, or approving a wrong address. Those can't be undone. Keep the phrase on paper offline, and read each review before you approve."]),
   ("has it been audited", ["I don't have any information about that, so I won't claim it. What I can tell you is how the app works: the key stays on your device in the Secure Enclave, and every move needs your approval."]),
   ("how much should i put in", ["That's your call, and I'd start small. Only put in what you can afford to lose while you get comfortable. Back up your phrase on paper before you add anything."])],
  w=1.4)

# More user phrasings (appended to the entries above so every entry has 12 to 20).
EXTRA_Q = {
    "aave_chains": ["what chains can aave be used on inside hermit", "is aave available on base in hermit", "does hermit let me use aave on optimism"],
    "aave_risks": ["is lending usdc on aave a safe way to earn", "what's the catch with aave yield", "could i lose my deposit on aave"],
    "aave_what": ["aave explained", "i keep seeing aave everywhere, what is it"],
    "aave_withdraw": ["can i withdraw only half of my aave position", "can i take out my whole aave deposit in one go", "do i need to withdraw everything from aave or can i do part",
                      "partial aave withdraw works?", "is it possible to pull only some funds out of aave", "can i withdraw from aave in pieces"],
    "ai_ask": ["does the ai ever need my private key", "is it normal for hermit to ask me to enter my seed after an update"],
    "ai_limits": ["can the ai go over my limit", "is there a max the assistant can send at once", "does the ai have a spending cap", "how do i stop the ai from spending too much"],
    "device_offline": ["does hermit need internet to answer", "can i use the assistant with no wifi", "will the chat work on the subway with no signal",
                       "is the model downloaded on my phone or online", "do i need to be online for the ai to reply"],
    "ens_send": ["does hermit accept ens names as the recipient"],
    "funding": ["what is the funding rate", "why is funding negative", "funding rate explained"],
    "il_example": ["what would impermanent loss be if one side goes {kx}"],
    "il_fees": ["do liquidity provider fees outweigh impermanent loss", "are lp fees enough to cover il", "how are lp fees paid out", "is impermanent loss worth the fees"],
    "il_realised": ["does the impermanent loss go away if i wait", "is il only on paper until i withdraw", "can price recovery undo impermanent loss"],
    "keys_enclave": ["what is a secure enclave anyway", "how does the iphone chip protect my wallet key"],
    "keys_icloud": ["will icloud restore my hermit wallet", "do i lose my wallet if i delete the app and restore from icloud"],
    "keys_passcode": ["can i approve with my passcode if the sensor fails", "is the passcode enough to use my wallet", "my sensor is broken, can i still approve"],
    "leverage": ["what does {lev}x leverage do to my risk", "how does leverage and margin work together"],
    "limit_order": ["limit or market for a swap", "what's a limit buy", "are limit orders better than market orders", "how does a limit order fill"],
    "liquidation": ["what is a liquidation in leveraged trading", "why do people get liquidated", "what price do i get liquidated at with {lev}x from an entry of ${entry}",
                    "how does liquidation work on hyperliquid"],
    "lost_no_phrase": ["phone died and i have no recovery phrase written anywhere", "never backed up my seed, what happens if my phone breaks", "can anyone get my funds back if i lost the phone and the phrase",
                       "i lost both my phone and my backup"],
    "new_phone": ["i'm replacing my phone, how do i keep my wallet", "how do i get hermit set up again on a new iphone", "do i need my seed phrase to switch phones", "restoring hermit on a new mac"],
    "perp_what": ["what is a perpetual swap", "perps vs futures"],
    "revoke": ["how do i cancel a token allowance", "what is revoke.cash"],
    "scam_alert": ["email says verify my hermit wallet or it gets locked", "someone told me my wallet will be deleted unless i confirm my seed", "is it true hermit can freeze wallets",
                   "dm says my wallet has a security issue and i must enter my words", "got a notification my wallet is on a watchlist, send funds to a safe address"],
    "scam_recovery": ["i got drained and someone in a group chat says he is a recovery specialist", "stranger offers to get my crypto back for 10 percent", "should i pay a recovery agent"],
    "seed_lost_paper": ["i lost my seed phrase backup but the app still opens", "my seed phrase paper is gone, wallet still works", "can hermit show me my seed phrase again, i lost the paper",
                        "i burned my paper with the words on it, can i get them again"],
    "slippage": ["why was my swap price worse than expected", "what does 1 percent slippage mean", "how does slippage hurt a big trade"],
    "swap_provider": ["what's the bridge under the hood", "who runs the cross chain routing here"],
}
for _f in FACTS:
    _f["q"] = _f["q"] + EXTRA_Q.get(_f["id"], [])

# A fourth (or more) hand-written variant for the entries that were written with three.
EXTRA_A = {
    "aave_chains": ["Aave v3 deposits work on Ethereum, Base and Arbitrum in Hermit. Tell me which one you want and I'll prepare it for you to approve with {BIO}."],
    "aave_risks": ["Lending on Aave can still lose money. A smart-contract bug, a drop in the price of what you deposited, or a stablecoin losing its peg can all hurt, and the rate is variable, so it can fall. Only deposit what you can accept losing."],
    "aave_what": ["Aave is a place to lend and borrow crypto. You deposit a token and earn interest paid by borrowers, at a variable rate. It's lending, not trading, so the risks are smart-contract, market and depeg risk."],
    "aave_withdraw": ["Yes, partial or full. Say how much you want out, or say all of it, and I'll prepare it. You approve the review with {BIO}."],
    "ai_limits": ["The assistant has a daily spend limit and a cap on each move. Beyond that, nothing goes out until you approve it with {BIO}. I don't know the exact amounts, so I won't guess."],
    "device_offline": ["Chatting works offline because the AI runs on your device. Prices, balances and sending a transaction need a connection."],
    "ens_send": ["Yes, you can send to a .eth name. Just type it in your message and I'll prepare the send for you to approve with {BIO}. Spell it exactly, because a lookalike name is a different address."],
    "funding": ["Funding is what keeps a perp's price close to the spot price: longs pay shorts when the rate is positive, and shorts pay longs when it's negative. With ${pos} open and a rate of {rate}%, that's about ${pay} each period."],
    "il_fees": ["Pools share trading fees with the people who supply liquidity, and those fees can cover impermanent loss or beat it, or fall short. High volume helps, big price moves hurt. Look at actual volume, not only the headline APR."],
    "il_realised": ["Not until you withdraw. While your liquidity sits in the pool, the loss versus holding is only on paper and moves with the price. If prices return to where they started, it disappears."],
    "keys_enclave": ["Your key is sealed inside the Secure Enclave, a protected chip in your device. Only {BIO} or your passcode unlocks it, and neither Hermit nor the AI can read it. That doesn't protect a written phrase that's lying around, so keep that offline."],
    "keys_passcode": ["{BIO} is the usual way, and your device passcode is the backup way to unlock the key. You still review and approve each move."],
    "leverage": ["Leverage is a multiplier on your position. With ${margin} at {lev}x you hold ${pos}, so a move of about {move}% against you can liquidate you, and a move in your favour pays {lev} times as much."],
    "limit_order": ["A limit order means only at this price or better. A market order means now, at whatever the price is. With a coin at ${now}, a limit buy at ${lim} only fills if it falls to ${lim}, and a market buy fills straight away."],
    "liquidation": ["Liquidation means your position is closed by force because the margin ran out. At {lev}x, a long from ${entry} is liquidated roughly near ${liq}, and slightly earlier in practice. Lower leverage means more distance."],
    "lost_no_phrase": ["No phrase, no wallet: that's how self-custody works. Hermit can't recover the key, so with the phone gone the funds are gone too. If the phone still works, move your funds to a new wallet and write down the phrase."],
    "new_phone": ["On the new device, choose restore and enter your {words}-word phrase, in Hermit or another wallet. The old phone's backup won't help, since the key isn't synced to iCloud. Check your balances before you wipe the old one."],
    "perp_what": ["A perp is a bet on a coin's price with no expiry. You can go long or short, often with leverage, and funding payments keep it near the spot price. It's riskier than holding the coin, because leverage can get you liquidated."],
    "revoke": ["Hermit has no approvals screen yet. To clean up old permissions, check your address on a tool like revoke.cash and revoke what you don't need. Each revoke is a transaction, so expect a small fee."],
    "scam_alert": ["Scam. A self-custody wallet can't be suspended, frozen or flagged, and nobody can ask you to verify with your {words} words. Don't click, don't reply, and don't move funds because a message says so."],
    "scam_recovery": ["Don't trust it, it's probably a second scam. Once a transaction is confirmed, nobody can undo it, and recovery agents who ask for a fee or your phrase are going after what's left. Keep your words to yourself."],
    "seed_lost_paper": ["Nobody can recover it, Hermit included, because only you ever had it. Since the wallet still opens, create a new wallet now, write its phrase on paper and move your funds across."],
    "seed_share": ["Don't share it. Anyone who has your {words} words controls your funds, and there's no undo. If someone needs access in an emergency, point them to a sealed paper copy in a safe place."],
    "slippage": ["Slippage is the price moving against you between the quote and the fill. With {slip}% on ${amt}, the cost can be about ${loss}. Bigger orders and thin markets make it worse."],
    "swap_provider": ["It's NEAR Intents for swaps and bridges in Hermit. A quote shows the amount you'd get before you approve anything with {BIO}."],
}
for _f in FACTS:
    _f["a"] = _f["a"] + EXTRA_A.get(_f["id"], [])

assert len(FACTS) == 45, len(FACTS)

# Required keywords for the LAST turn of eval items that end on follow-up #3 (index 2). Every follow-up answer must match.
FU3_REQ = {
    "seed_where": [r"digital|offline|paper"], "seed_screenshot": [r"offline|paper"], "seed_cloud": [r"second|copy"], "seed_type_site": [r"fake|scam"],
    "seed_share": [r"control"], "seed_what_is": [r"can'?t be edited|new wallet"], "seed_lost_paper": [r"gone", r"phrase"],
    "keys_store": [r"can'?t", r"phrase"], "keys_icloud": [r"gone"], "keys_enclave": [r"scam"], "keys_passcode": [r"phrase"],
    "ai_see": [r"new wallet|exposed"], "ai_ask": [r"new wallet"], "scam_support": [r"block"], "scam_recovery": [r"scam"], "scam_alert": [r"self-custody"],
    "lost_phone": [r"gone"], "new_phone": [r"key|wallet"], "lost_no_phrase": [r"secure enclave"],
    "il_what": [r"\bno\b", r"pool"], "il_lp": [r"smart.contract"], "il_realised": [r"pool|liquidity", r"loss"], "il_fees": [r"fee", r"volume|price"],
    "il_example": [r"fee", r"hold"], "aave_what": [r"variable", r"\bno\b"], "aave_il": [r"smart.contract"], "aave_risks": [r"withdraw"],
    "swap_route": [r"near intents", r"bridge"], "swap_provider": [r"near intents"], "approve_all": [r"approv", r"each|one"],
    "ai_limits": [r"scam|reverse"], "device_private": [r"device|phone"], "device_offline": [r"device|model"], "ens_send": [r"typo|letter"],
    "aave_chains": [r"fee"], "aave_withdraw": [r"everything|all"], "chains_list": [r"one hermit wallet"], "perp_what": [r"hyperliquid"],
    "leverage": [r"liquidat"], "funding": [r"funding|rate"], "liquidation": [r"position"], "slippage": [r"quote"], "limit_order": [r"prepare", r"approv"],
    "revoke": [r"revoke\.cash|approvals"], "hermit_safe": [r"paper|phrase|small"],
}
assert set(FU3_REQ) == {f["id"] for f in FACTS}
for _f in FACTS:
    _f["fu_req"] = FU3_REQ[_f["id"]]

# ------------------------------------------------------------------------------------------------ text machinery
PLACE_RE = re.compile(r"\{(\w+)\}")
BASE_SLOTS = {"words": str(WORDS)}
PROTECT = {"eth", "usdc", "usdt", "sol", "btc", "bnb", "aave", "uniswap", "hermit", "near", "base", "arbitrum", "polygon", "monad", "solana",
           "ethereum", "dropbox", "telegram", "whatsapp", "icloud", "revoke", "cash", "ens", "lp", "il", "amm", "dapp", "dapps", "defi", "ai"}


def fill(tpl, d):
    return PLACE_RE.sub(lambda m: str(d[m.group(1)]) if m.group(1) in d else m.group(0), tpl)


def placeholders(tpl):
    return set(PLACE_RE.findall(tpl))


def typo(rng, text):
    """Typo layer for user text. Skips placeholders, numbers and tickers."""
    toks = text.split(" ")
    ok = [i for i, t in enumerate(toks) if re.fullmatch(r"[a-z]{5,}", t) and t not in PROTECT]
    kind = rng.choice(["swap", "vowel", "double", "pls", "rn", "punct", "lower"])
    if kind == "swap" and ok:
        i = rng.choice(ok)
        t = toks[i]
        j = rng.randrange(1, len(t) - 2)
        toks[i] = t[:j] + t[j + 1] + t[j] + t[j + 2:]
    elif kind == "vowel" and ok:
        i = rng.choice(ok)
        t = toks[i]
        vs = [k for k, c in enumerate(t) if c in "aeiou" and 0 < k < len(t) - 1]
        if vs:
            k = rng.choice(vs)
            toks[i] = t[:k] + t[k + 1:]
    elif kind == "double" and ok:
        i = rng.choice(ok)
        t = toks[i]
        k = rng.randrange(1, len(t) - 1)
        toks[i] = t[:k] + t[k] + t[k:]
    elif kind == "pls":
        return rng.choice(["pls ", "plz "]) + text if rng.random() < 0.5 else text + rng.choice([" pls", " plz"])
    elif kind == "rn":
        return text + " rn"
    elif kind == "punct":
        return text.rstrip("?.!")
    elif kind == "lower":
        return text.lower()
    return " ".join(toks)


def style(rng, text):
    """Casing / punctuation variety on user text (after placeholders are filled)."""
    r = rng.random()
    if r < 0.02:
        return text.upper()
    if r < 0.30:
        text = text[:1].lower() + text[1:] if text[:2] not in ("I ", "I'") else text
    if rng.random() < 0.20:
        text = text.rstrip("?.!")
    return text


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def n_sentences(t):
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'$0-9(])", t.strip()) if s])


NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
PRODUCT_NUM = re.compile(r"\bv3\b|\b0x\w*")


def nums(text):
    out = set()
    for m in NUM_RE.findall(PRODUCT_NUM.sub("", text)):
        m = m.rstrip(",").replace(",", "")
        if m:
            out.add(Decimal(m))
    return out


def with_user_safe(rng, base, fact):
    for _ in range(8):
        s = H.with_user(base, rng)
        if "5x" in s and fact["grp"] == 14:
            continue
        return s
    return base


def system_for(rng, fact, bio):
    s = with_user_safe(rng, SYSTEM_V2, fact)
    return s.replace("{BIO}", bio)


def answer_ok(text, req, extra_bad=()):
    """Return list of problems for one rendered answer."""
    probs = []
    w = len(text.split())
    if w > 90:
        probs.append(f"{w} words")
    ns = n_sentences(text)
    if not 2 <= ns <= 4:
        probs.append(f"{ns} sentences")
    for rx in BAD_RE:
        if rx.search(text):
            probs.append("bad:" + rx.pattern[:40])
    for p in extra_bad:
        if re.search(p, text, re.I):
            probs.append("fact-bad:" + p[:40])
    for g in req:
        if not re.search(g, text, re.I):
            probs.append("missing:" + g[:40])
    if "{" in text or "}" in text:
        probs.append("unfilled placeholder")
    return probs


# ------------------------------------------------------------------------------------------------ static self-check of the table
def check_table():
    errs = []
    for f in FACTS:
        if not 12 <= len(f["q"]) <= 20:
            errs.append(f"{f['id']}: {len(f['q'])} phrasings")
        if not 4 <= len(f["a"]) <= 8:
            errs.append(f"{f['id']}: {len(f['a'])} variants")
        if len(f["fu"]) != 3:
            errs.append(f"{f['id']}: {len(f['fu'])} follow-ups")
        if len({norm(q) for q in f["q"]}) != len(f["q"]):
            errs.append(f"{f['id']}: duplicate phrasing")
        for tpl in f["a"] + [a for _, al in f["fu"] for a in al]:
            raw = PRODUCT_NUM.sub("", PLACE_RE.sub("", tpl))
            if re.search(r"\d", raw):
                errs.append(f"{f['id']}: raw digit in answer {tpl[:50]}")
    return errs


# ------------------------------------------------------------------------------------------------ row builders
def quotas(rng, n_total, pools):
    """Largest-remainder quotas by weight, capped by each entry's available phrasings."""
    ws = {f["id"]: f["w"] for f in FACTS}
    tot = sum(ws.values())
    raw = {k: n_total * w / tot for k, w in ws.items()}
    q = {k: min(int(v), len(pools[k])) for k, v in raw.items()}
    rem = sorted(raw, key=lambda k: (raw[k] - int(raw[k]), rng.random()), reverse=True)
    while sum(q.values()) < n_total:
        moved = False
        for k in rem:
            if sum(q.values()) >= n_total:
                break
            if q[k] < len(pools[k]):
                q[k] += 1
                moved = True
        if not moved:
            break
    return q


def build_example(rng, fact, qi, bio, fu_idx, ev, seen):
    """One record. qi = index of the phrasing, fu_idx = follow-up index or None."""
    d = dict(BASE_SLOTS)
    if fact["slots"]:
        d.update(fact["slots"](rng, ev))
    d["BIO"] = bio
    for _ in range(12):
        qt = fact["q"][qi]
        if rng.random() < 0.15:
            qt = typo(rng, qt)
        u1 = style(rng, fill(qt, d))
        if norm(u1) not in seen:
            break
    else:
        return None
    a1 = fill(rng.choice(fact["a"]), d)
    msgs = [{"role": "system", "content": system_for(rng, fact, bio)}, {"role": "user", "content": u1}, {"role": "assistant", "content": a1}]
    users, answers, reqs = [u1], [a1], [fact["req"]]
    if fu_idx is not None:
        fut, fua = fact["fu"][fu_idx]
        u2 = style(rng, fill(fut, d))
        if norm(u2) in seen or norm(u2) == norm(u1):
            return None
        a2 = fill(rng.choice(fua), d)
        msgs += [{"role": "user", "content": u2}, {"role": "assistant", "content": a2}]
        users.append(u2)
        answers.append(a2)
        reqs.append(fact["fu_req"] if fu_idx == 2 else [])
    meta = {"source": "knowledge_fix", "generator": "knowledge_fix", "kind": "knowledge_fix", "fact": fact["id"], "group": fact["grp"],
            "phrasing": qi, "followup": fu_idx, "bio": bio, "world": None, "world_final": None, "expect": [{"text": True}],
            "checks": ["facts_ok", "restraint"], "slots": {k: v for k, v in d.items() if k not in ("BIO",)}, "lang": "en"}
    return {"messages": msgs, "tools": True, "meta": meta}


def facts_ok_spec(fact, final_turn_has_fu):
    """Checks for the LAST assistant turn: the fact's keyword groups, or the follow-up's groups; plus the bad-claim regexes."""
    return {"fact": fact["id"], "required": fact["fu_req"] if final_turn_has_fu else fact["req"], "forbidden": BAD + fact["bad"], "ignore_case": True}


def gen_train(seed, n):
    rng = random.Random(seed)
    seen = set()
    pools = {f["id"]: [i for i in range(len(f["q"])) if i % 5 != 0] for f in FACTS}
    qt = quotas(rng, n, pools)
    jobs = []
    for f in FACTS:
        idxs = rng.sample(pools[f["id"]], qt[f["id"]])
        fus = [0, 1]
        rng.shuffle(fus)
        for qi in idxs:
            fu = fus.pop() if fus and rng.random() < 0.5 else None
            jobs.append((f, qi, fu))
    rng.shuffle(jobs)
    k = round(len(jobs) * 0.70)
    bios = ["Face ID"] * k + ["Touch ID"] * (len(jobs) - k)
    rng.shuffle(bios)
    rows = []
    used = {(f["id"], qi) for f, qi, _ in jobs}
    for (f, qi, fu), bio in zip(jobs, bios):
        crng = random.Random(rng.getrandbits(64))
        ex = build_example(crng, f, qi, bio, fu, False, seen)
        spare = [j for j in pools[f["id"]] if (f["id"], j) not in used]
        while ex is None and spare:                     # collision: take another unused phrasing of the same fact
            qj = spare.pop(crng.randrange(len(spare)))
            used.add((f["id"], qj))
            ex = build_example(crng, f, qj, bio, fu, False, seen)
        if ex is None:
            continue
        for m in ex["messages"]:
            if m["role"] == "user":
                seen.add(norm(m["content"]))
        rows.append(ex)
    # top up to exactly n from unused phrasings (collisions can drop a few)
    spare_jobs = [(f, j) for f in FACTS for j in pools[f["id"]] if (f["id"], j) not in used]
    rng.shuffle(spare_jobs)
    while len(rows) < n and spare_jobs:
        f, qj = spare_jobs.pop()
        crng = random.Random(rng.getrandbits(64))
        ex = build_example(crng, f, qj, "Face ID" if rng.random() < 0.7 else "Touch ID", None, False, seen)
        if ex is None:
            continue
        for m in ex["messages"]:
            if m["role"] == "user":
                seen.add(norm(m["content"]))
        rows.append(ex)
    return rows


def gen_eval(seed, n=40):
    rng = random.Random(seed)
    # one entry per topic group first, then fill from the rest; all distinct
    by_grp = {}
    for f in FACTS:
        by_grp.setdefault(f["grp"], []).append(f)
    chosen = [rng.choice(v) for v in by_grp.values()]
    rest = [f for f in FACTS if f not in chosen]
    rng.shuffle(rest)
    chosen += rest[: n - len(chosen)]
    rng.shuffle(chosen)
    k = round(n * 0.70)
    bios = ["Face ID"] * k + ["Touch ID"] * (n - k)
    rng.shuffle(bios)
    mt = set(rng.sample(range(n), max(1, round(n * 0.2))))
    items, seen = [], set()
    for i, (f, bio) in enumerate(zip(chosen, bios)):
        crng = random.Random(rng.getrandbits(64))
        qi = crng.choice([j for j in range(len(f["q"])) if j % 5 == 0])
        ex = build_example(crng, f, qi, bio, 2 if i in mt else None, True, seen)
        if ex is None:
            continue
        for m in ex["messages"]:
            if m["role"] == "user":
                seen.add(norm(m["content"]))
        msgs = ex["messages"]
        last_user = max(j for j, m in enumerate(msgs) if m["role"] == "user")
        fu = ex["meta"]["followup"]
        items.append({"id": f"knowledge_fix-{i:02d}", "kind": "knowledge_fix", "fact": f["id"], "tools": True, "bio": bio,
                      "context": msgs[: last_user + 1], "reference": msgs[last_user + 1:], "expect": [{"text": True}], "world": None,
                      "checks": ["facts_ok", "restraint"], "facts_ok": facts_ok_spec(f, fu is not None),
                      "gold": {"calls": [], "answer_checks": facts_ok_spec(f, fu is not None)}, "meta": ex["meta"]})
    return items


# ------------------------------------------------------------------------------------------------ self-check
def selfcheck(train, ev):
    errs = []
    errs += check_table()
    # every variant (and follow-up) of every fact, over several slot draws and both device words
    rng = random.Random(12345)
    for f in FACTS:
        for bio in ("Face ID", "Touch ID"):
            for trial in range(6):
                d = dict(BASE_SLOTS, BIO=bio)
                if f["slots"]:
                    d.update(f["slots"](rng, trial % 2 == 1))
                for tpl in f["a"]:
                    t = fill(tpl, d)
                    p = answer_ok(t, f["req"], f["bad"])
                    other = "Touch ID" if bio == "Face ID" else "Face ID"
                    if other.lower() in t.lower():
                        p.append("other bio word")
                    if p:
                        errs.append(f"variant {f['id']}: {p}: {t[:60]}")
                for fi, (fut, fua) in enumerate(f["fu"]):
                    for tpl in fua:
                        t = fill(tpl, d)
                        p = answer_ok(t, f["fu_req"] if fi == 2 else [], f["bad"])
                        if p:
                            errs.append(f"followup {f['id']}: {p}: {t[:60]}")
                for tpl in f["q"] + [x for x, _ in f["fu"]]:
                    t = fill(tpl, d)
                    if "{" in t:
                        errs.append(f"user placeholder unfilled {f['id']}: {t}")
    if errs:
        return sorted(set(errs))[:60]
    by_id = {f["id"]: f for f in FACTS}
    seen_user, tot_tokens = {}, 0
    for name, rows in (("train", train), ("eval", ev)):
        for i, r in enumerate(rows):
            rec = r if name == "train" else {"messages": r["context"] + r["reference"], "tools": r["tools"], "meta": r["meta"]}
            tag = f"{name}[{i}]"
            try:
                rec = json.loads(json.dumps(rec, ensure_ascii=False))
            except Exception as e:  # noqa
                errs.append(f"{tag}: not json {e}")
                continue
            meta = rec["meta"]
            f = by_id[meta["fact"]]
            msgs = rec["messages"]
            bio = meta["bio"]
            other = "Touch ID" if bio == "Face ID" else "Face ID"
            if meta["source"] != "knowledge_fix" or rec["tools"] is not True:
                errs.append(f"{tag}: bad meta/tools")
            roles = [m["role"] for m in msgs]
            if roles[0] != "system" or roles[1:] != ["user", "assistant"] * ((len(roles) - 1) // 2):
                errs.append(f"{tag}: role order {roles}")
            sysm = msgs[0]["content"]
            if not sysm.startswith(SYSTEM_V2.replace("{BIO}", bio)[:200]) or "{BIO}" in sysm:
                errs.append(f"{tag}: system prompt")
            if bio not in sysm:
                errs.append(f"{tag}: bio word missing from system")
            allowed = set()
            for m in msgs:
                if m["role"] == "assistant" and (m.get("tool_calls") or "<tool_call>" in m["content"]):
                    errs.append(f"{tag}: tool call present")
                if m["role"] == "tool":
                    errs.append(f"{tag}: tool message")
                if other.lower() in m["content"].lower():
                    errs.append(f"{tag}: other bio word in {m['role']}")
                if "{BIO}" in m["content"] or "Face ID" in m["content"] and bio != "Face ID":
                    errs.append(f"{tag}: BIO leak")
            for k, v in meta["slots"].items():
                allowed |= nums(str(v))
            asst = [m["content"] for m in msgs if m["role"] == "assistant"]
            users = [m["content"] for m in msgs if m["role"] == "user"]
            for u in users:
                allowed |= nums(u)
            for j, a in enumerate(asst):
                req = f["req"] if j == 0 else (f["fu_req"] if meta["followup"] == 2 else [])
                # the last turn of an eval item with follow-up has no keyword set; first answers always do
                p = answer_ok(a, req, f["bad"])
                if p:
                    errs.append(f"{tag}: answer {j} {p}")
                extra = nums(a) - allowed
                if extra:
                    errs.append(f"{tag}: unbacked numbers {sorted(extra)} in {a[:50]}")
                if re.search(r"0x[0-9a-fA-F]{40}", a) and not any(re.search(r"0x[0-9a-fA-F]{40}", u) for u in users):
                    errs.append(f"{tag}: address in answer")
            # round trip through the lab renderer: full 19-tool block, no call in the conversation part
            text = H.render_text(msgs, True)
            if "# Tools" not in text or len(re.findall(r'\n\{"type": "function"', text)) != len(TOOL_NAMES):
                errs.append(f"{tag}: tool block is not the 19-tool v2 block")
            head, sep, conv = text.partition("</tool_call><|im_end|>\n")
            calls_, bad_ = H.parse_calls(conv)
            if not sep or "<tool_call>" in conv or calls_ or bad_:
                errs.append(f"{tag}: a tool call shows up in the rendered conversation")
            for a_ in asst:
                if a_ + "<|im_end|>" not in conv:
                    errs.append(f"{tag}: answer missing from render")
            tot_tokens = max(tot_tokens, len(text) // 3)
            for u in users:
                key = norm(u)
                if key in seen_user:
                    errs.append(f"{tag}: duplicate user text (also {seen_user[key]}): {u[:50]}")
                seen_user[key] = tag
    # eval vs train disjointness, 5-gram overlap
    def grams(t):
        w = norm(t).split()
        return {tuple(w[i:i + 5]) for i in range(max(0, len(w) - 4))}
    tr_users = [m["content"] for r in train for m in r["messages"] if m["role"] == "user"]
    tr_g = [(u, grams(u)) for u in tr_users]
    for r in ev:
        for m in r["context"]:
            if m["role"] != "user":
                continue
            g = grams(m["content"])
            if not g:
                continue
            for u, gu in tr_g:
                if gu and len(g & gu) / len(g | gu) >= 0.8:
                    errs.append(f"{r['id']}: near-duplicate of train text: {m['content'][:40]} ~ {u[:40]}")
                    break
    return errs[:60], tot_tokens


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--eval-seed", type=int, default=9310)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--out", default=OUT_TRAIN)
    ap.add_argument("--out-eval", default=OUT_EVAL)
    a = ap.parse_args()
    train = gen_train(a.seed, a.n)
    ev = gen_eval(a.eval_seed, a.n_eval)
    res = selfcheck(train, ev)
    if isinstance(res, list):
        res = (res, 0)
    errs, maxtok = res
    il_variants = [(t, "{expr}" in t) for f in FACTS if f["grp"] == 6 for t in f["a"]]
    frac = sum(1 for _, x in il_variants if x) / len(il_variants)
    stats = {"train": len(train), "eval": len(ev), "facts": len(FACTS), "il_example_variant_share": round(frac, 3),
             "face_id_train": sum(1 for r in train if r["meta"]["bio"] == "Face ID"), "multiturn_train": sum(1 for r in train if r["meta"]["followup"] is not None),
             "multiturn_eval": sum(1 for r in ev if r["meta"]["followup"] is not None), "max_tokens_est": maxtok}
    if errs:
        print("self-check FAILED:", file=sys.stderr)
        for e in errs:
            print("  ", e, file=sys.stderr)
        print(json.dumps(stats), file=sys.stderr)
        print(f"knowledge_fix: self-check failed ({len(errs)} problems)")
        sys.exit(1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.out_eval), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out_eval, "w") as fh:
        for r in ev:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter(r["meta"]["fact"] for r in train)
    print(json.dumps(stats), file=sys.stderr)
    print("per-fact rows:", dict(sorted(kinds.items())), file=sys.stderr)
    print(f"knowledge_fix: {len(train)} train, {len(ev)} eval, self-check ok")


if __name__ == "__main__":
    main()
