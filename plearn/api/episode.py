"""The episode contract and the agent's turn.

An environment class (a task's 'env') implements:

    @classmethod specs(split, n, seed, **env_args)    the first n episode specs of a split (JSON values, the env's own order)
    __init__(spec, **env_args)                         one episode
    done                                               True once the episode has ended
    messages(carry)                                    (system text, user parts, pictures) of the next turn: the parts are a
                                                       string, or a list of {"type": "text", "text": ...} and {"type": "image"};
                                                       carry: show the agent's belief line of its last turn
    step(belief, action, record)                       apply the agent's two lines; record: the turn's facts to keep
    row()                                              the episode's result: 'correct' and the task's metric fields
    close()                                            free the episode's processes

The agent's turn (RLVB (EM)'s turn format, the same in training and evaluation): its reasoning (stopped at
</think>, at most max_think tokens; a cut reasoning gets the closing </think>), then the belief line, then the action line,
each the first line of its own completion. The one-line turn (one_line, the action-only test of a model, for example
the base model's "Base" row): the reasoning, then the action line alone (action_line of one completion); the belief line is empty.
"""
from __future__ import annotations
import os

CLOSE, TERM = "\n</think>\n\n", "\n"                                 # the trainer's turn protocol (plearn/method/trainer.py)
CARRY_MARK = "Your belief line after your last turn: "         # the carried line's prefix (the analysis scripts read the line after it)
CARRY_RULE = ("This line is your own earlier estimate, not evidence. Where the evidence shown here disagrees with it, the evidence "
              "decides: correct the line.")


def carried(note):
    """The agent's own belief line of its last turn, as every prompt of the agent's turn format carries it, in training and in
    the evaluations: the line, then the rule that the evidence overrules it (beliefs can be updated and must be
    subject to correction; observed evidence has greater authority than the beliefs the model wrote before). '' with no line."""
    return f"\n{CARRY_MARK}{note}\n{CARRY_RULE}" if note else ""


def first_line(t): return t.strip().split("\n")[0].strip()


def ordered(belief, action):
    """The two lines as the trainer reads them: an ACTION line first and a non-ACTION line second are swapped."""
    return (action, belief) if belief.upper().startswith("ACTION:") and not action.upper().startswith("ACTION:") else (belief, action)


def action_line(reply):
    """The action of a one-line reply: its first line that starts with ACTION:, else its first line."""
    for x in reply.splitlines():
        if x.strip().upper().startswith("ACTION:"): return x.strip()
    return first_line(reply)


def thought_rec(z):
    """The turn record of a thought: its length and whether the cap cut it. PL_SAVE_THOUGHTS=1 (to check the thinking
    content, for example for repeated sentences): also the thought text itself (key "thought"), for a thought sample; off by default (the rows
    stay small)."""
    rec = {"thought_chars": len(z["text"]), "thought_cut": z["finish"] != "stop"}
    if os.environ.get("PL_SAVE_THOUGHTS") == "1": rec["thought"] = z["text"]
    return rec


def turn(srv, prompt, g, use_adapter=True, priority=0, one_line=False):
    """One turn of the agent at a tagged prompt: (reasoning, belief line, action line, record). one_line: the one-line turn
    (the module text), with an empty belief line."""
    z = srv.complete([prompt], n=1, max_tokens=g["max_think"], temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"],
                     stop=["</think>"], use_adapter=use_adapter, priority=priority)[0][0]
    pre = prompt + z["text"] + CLOSE
    if one_line:
        a = action_line(srv.complete([pre], n=1, max_tokens=g["max_action"], temperature=g["line_temperature"], top_p=1.0, top_k=-1,
                                     use_adapter=use_adapter, priority=priority)[0][0]["text"])
        return z["text"], "", a, thought_rec(z)
    b = first_line(srv.complete([pre], n=1, max_tokens=g["max_belief"], temperature=g["line_temperature"], top_p=1.0, top_k=-1,
                                use_adapter=use_adapter, priority=priority)[0][0]["text"])
    a = first_line(srv.complete([pre + b + TERM], n=1, max_tokens=g["max_action"], temperature=g["line_temperature"], top_p=1.0, top_k=-1,
                                use_adapter=use_adapter, priority=priority)[0][0]["text"])
    b, a = ordered(b, a)
    return z["text"], b, a, thought_rec(z)


def submit_turn(srv, prompt, g, belief, use_adapter=True, priority=0):
    """The turn for the SUBMIT alone, with no new thought (RoboQuest final_submit=prefill: the SUBMIT needs no
    new thinking, it only follows the last turn): the thought is empty, the belief line is the agent's own last one (as
    carried), and the action line is started as 'ACTION: SUBMIT'; the model writes only the answer after it. Returns the same
    four values as turn()."""
    b = first_line(belief) if belief else ""
    head = "ACTION: SUBMIT"
    txt = srv.complete([prompt + CLOSE + (b + TERM if b else "") + head], n=1, max_tokens=g["max_action"], temperature=g["line_temperature"],
                       top_p=1.0, top_k=-1, use_adapter=use_adapter, priority=priority)[0][0]["text"]
    rest = txt.strip("\n").split("\n")[0].strip() if txt.strip() else ""
    return "", b, (head + " " + rest).strip(), {"thought_chars": 0, "thought_cut": False, "submit_prefill": True}

