"""Arquitetura da PolicyNet (ResNet + Squeeze-and-Excitation, 1.968 logits de acao) e helpers de inferencia."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kairos.chess_ext import (
    ACTION_FROM,
    ACTION_SIZE,
    ACTION_TO,
    INPUT_CHANNELS,
    MOVE_TAG_NAMES,
    PHASE_NAMES,
    THEME_TAG_NAMES,
    board_to_planes,
    legal_action_ids,
)

N_THEMES = len(THEME_TAG_NAMES)     # 9
N_MOVE_TYPES = len(MOVE_TAG_NAMES)  # 4


class SE(nn.Module):

    def __init__(self, c, ratio=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(8, c // ratio))
        self.fc2 = nn.Linear(max(8, c // ratio), c)

    def forward(self, x):
        s = x.mean((2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class ResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.se = SE(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.se(self.b2(self.c2(h)))
        return F.relu(x + h)


class PolicyNet(nn.Module):

    def __init__(self, channels=128, blocks=10):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # policy principal (1.968 saidas)
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(32 * 64, ACTION_SIZE)

        # from/to por casa: somam no logit de cada lance
        self.ft = nn.Conv2d(channels, 2, 1)

        # casas de origem/destino legais (regras)
        self.legal_sq = nn.Conv2d(channels, 2, 1)

        # cabecas globais
        self.g_fc = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(inplace=True))
        self.g_phase = nn.Linear(128, len(PHASE_NAMES))
        self.g_mtype = nn.Linear(128, N_MOVE_TYPES)
        self.g_theme = nn.Linear(128, N_THEMES)

        self.register_buffer("a_from", torch.as_tensor(ACTION_FROM), persistent=False)
        self.register_buffer("a_to", torch.as_tensor(ACTION_TO), persistent=False)

    def forward(self, x):

        h = self.tower(self.stem(x))

        B = h.shape[0]

        ft = self.ft(h).flatten(2)          # [B, 2, 64]
        from_logit, to_logit = ft[:, 0], ft[:, 1]

        policy = (
            self.p_fc(self.p_conv(h).flatten(1))
            + from_logit[:, self.a_from]
            + to_logit[:, self.a_to]
        )

        g = self.g_fc(h.mean((2, 3)))

        return {
            "policy": policy,
            "from": from_logit,
            "to": to_logit,
            "legal_sq": self.legal_sq(h).flatten(2),   # [B, 2, 64]
            "phase": self.g_phase(g),
            "mtype": self.g_mtype(g),
            "theme": self.g_theme(g),
        }


# ============================================================
# INFERENCE HELPERS (para duelos / diagnosticos)
# ============================================================

def load_policy(path, device="cpu"):

    ck = torch.load(path, map_location=device, weights_only=False)

    model = PolicyNet(**ck["config"])
    model.load_state_dict(ck["model"])

    return model.to(device).eval()


@torch.no_grad()
def choose_move(model, board, device="cpu", temperature=0.0):

    x = torch.from_numpy(board_to_planes(board)).float()[None].to(device)

    logits = model(x)["policy"][0].float().cpu()

    ids = legal_action_ids(board)

    scores = logits[ids]

    if temperature <= 0:
        i = int(scores.argmax())
    else:
        probs = torch.softmax(scores / temperature, 0)
        i = int(torch.multinomial(probs, 1))

    return list(board.legal_moves)[i]
