"""Synthetic support-ticket data: train / held-out split with DIFFERENT phrasings."""
import json, random
from pathlib import Path
import sys

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
random.seed(7)

TOPICS = {
    "billing": {
        "train": ["I was charged twice for my subscription", "my invoice shows the wrong amount", "I need a refund for last month",
                  "the promo code did not apply at checkout", "my card was billed after I cancelled",
                  "there is an unknown fee on my statement", "please update the payment method on my account",
                  "I was overcharged for the annual plan"],
        "test": ["a duplicate payment appeared on my bank statement", "can you reverse a charge I never approved",
                 "my receipt total does not match what I paid", "I want my money back for the unused credits"],
    },
    "login": {
        "train": ["I cannot reset my password", "the verification code never arrives", "my account is locked after too many tries",
                  "I forgot the email I signed up with", "two factor login keeps failing",
                  "the sign in page just spins forever", "I am logged out every few minutes",
                  "my password reset link has expired"],
        "test": ["I can't get into my account anymore", "the login screen says my credentials are invalid",
                 "I never received the sign in email", "my authenticator app codes are rejected"],
    },
    "shipping": {
        "train": ["my package has not arrived yet", "the tracking number shows no updates", "I received the wrong item",
                  "the parcel arrived damaged", "my order was delivered to a different address",
                  "delivery is two weeks late", "part of my order is missing from the box",
                  "I need to change the delivery address"],
        "test": ["my order still hasn't shown up", "the courier marked it delivered but I have nothing",
                 "the box was crushed when it got here", "I got someone else's package"],
    },
    "bug": {
        "train": ["the app crashes when I upload a photo", "the page shows an error when I click save",
                  "the dashboard will not load any charts", "buttons stop responding after the update",
                  "the export feature produces an empty file", "notifications are showing twice",
                  "the search returns nothing for any query", "the mobile app freezes on the home screen"],
        "test": ["everything freezes when I open settings", "I get a blank screen after pressing submit",
                 "the app quits as soon as I start a video", "the report download fails with an error"],
    },
}
URGENT = ["", "", " This is urgent.", " Please fix this asap.", " I need this resolved immediately."]

def make(desc):
    suffix = random.choice(URGENT)
    urg = "high" if suffix else "low"
    return f"Ticket: {desc}.{suffix}\nAnswer:", urg

train, test = [], []
for dept, d in TOPICS.items():
    for desc in d["train"]:
        for _ in range(6):
            prompt, urg = make(desc)
            train.append({"text": f"{prompt} {{\"department\": \"{dept}\", \"urgency\": \"{urg}\"}}"})
    for desc in d["test"]:
        for _ in range(3):
            prompt, urg = make(desc)
            test.append({"prompt": prompt, "department": dept, "urgency": urg})
random.shuffle(train); random.shuffle(test)
(out / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
(out / "heldout.jsonl").write_text("\n".join(json.dumps(r) for r in test) + "\n")
print(f"train={len(train)} heldout={len(test)}")
