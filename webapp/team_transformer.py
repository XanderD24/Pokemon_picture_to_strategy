"""Inference wrapper for the masked-team transformer.

Loads the trained checkpoint (state_dict + config + vocabs) plus the raw-data-vector
(.pkl) dicts, then predicts the full set (species/ability/item/4 moves) of a 6th
Pokemon given up to 5 known team members.

The model class mirrors `train_masked_team_transformer_colab.ipynb` (cells 12, 16, 24)
— including the legality buffers and the legality/team-rule-constrained `predict`
decoder — but is driven by the checkpoint's `config`/`vocabs` rather than notebook
globals, so it works regardless of which preset was trained. The legality buffers are
restored from the checkpoint, so POKEMON_DICT is not needed at inference time.
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo layout: this file lives in webapp/, artifacts live under the repo root.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "Models" / "masked_team_transformer.pt"
DEFAULT_RDV_DIR = ROOT / "Data" / "Transformer Ready Vectors"
DEFAULT_TEAM_DIR = ROOT / "Data" / "Scraped Pokemon Teams"

# Decoding settings (match the trained notebook's Section-6 config). With top-k/p
# disabled, decoding is deterministic argmax with legality enforcement applied.
ENFORCE_LEGALITY = True
TOP_K = 0
TOP_P = 0
TEMP = 1.0
SAMPLING_PHASE = "both"


# --------------------------------------------------------------------------- #
# Mega-stone handling (notebook cell 12)
# --------------------------------------------------------------------------- #
MEGA_STONE_TO_FORM = {
    "Abomasite": "Abomasnow-Mega", "Absolite": "Absol-Mega",
    "Aerodactylite": "Aerodactyl-Mega", "Aggronite": "Aggron-Mega",
    "Alakazite": "Alakazam-Mega", "Altarianite": "Altaria-Mega",
    "Ampharosite": "Ampharos-Mega", "Audinite": "Audino-Mega",
    "Banettite": "Banette-Mega", "Beedrillite": "Beedrill-Mega",
    "Blastoisinite": "Blastoise-Mega", "Cameruptite": "Camerupt-Mega",
    "Chandelurite": "Chandelure-Mega",
    "Charizardite X": "Charizard-Mega-X", "Charizardite Y": "Charizard-Mega-Y",
    "Chesnaughtite": "Chesnaught-Mega", "Chimechite": "Chimecho-Mega",
    "Clefablite": "Clefable-Mega", "Crabominite": "Crabominable-Mega",
    "Delphoxite": "Delphox-Mega", "Dragoninite": "Dragonite-Mega",
    "Drampanite": "Drampa-Mega", "Emboarite": "Emboar-Mega",
    "Excadrite": "Excadrill-Mega", "Feraligite": "Feraligatr-Mega",
    "Floettite": "Floette-Mega", "Froslassite": "Froslass-Mega",
    "Galladite": "Gallade-Mega", "Garchompite": "Garchomp-Mega",
    "Gardevoirite": "Gardevoir-Mega", "Gengarite": "Gengar-Mega",
    "Glalitite": "Glalie-Mega", "Glimmoranite": "Glimmora-Mega",
    "Golurkite": "Golurk-Mega", "Greninjite": "Greninja-Mega",
    "Gyaradosite": "Gyarados-Mega", "Hawluchanite": "Hawlucha-Mega",
    "Heracronite": "Heracross-Mega", "Houndoominite": "Houndoom-Mega",
    "Kangaskhanite": "Kangaskhan-Mega", "Lopunnite": "Lopunny-Mega",
    "Lucarionite": "Lucario-Mega", "Manectite": "Manectric-Mega",
    "Medichamite": "Medicham-Mega", "Meganiumite": "Meganium-Mega",
    "Meowsticite": "Meowstic-Mega", "Pidgeotite": "Pidgeot-Mega",
    "Pinsirite": "Pinsir-Mega", "Sablenite": "Sableye-Mega",
    "Scizorite": "Scizor-Mega", "Scovillainite": "Scovillain-Mega",
    "Sharpedonite": "Sharpedo-Mega", "Skarmorite": "Skarmory-Mega",
    "Slowbronite": "Slowbro-Mega", "Starminite": "Starmie-Mega",
    "Steelixite": "Steelix-Mega", "Tyranitarite": "Tyranitar-Mega",
    "Venusaurite": "Venusaur-Mega", "Victreebelite": "Victreebel-Mega",
}


# --------------------------------------------------------------------------- #
# Model (notebook cell 24) — driven by checkpoint config + vocab sizes.
# Legality buffers are registered with the right shapes and restored from the
# checkpoint's state_dict (no POKEMON_DICT needed at inference).
# --------------------------------------------------------------------------- #
class MaskedTeamTransformer(nn.Module):
    def __init__(self, config, n_species, n_ability, n_item, n_move):
        super().__init__()
        self.emb = config["EMB"]
        self.move_rdv_dim = config["MOVE_RDV_DIM"]

        self.species_emb = nn.Embedding(n_species, self.emb["species"])
        self.ability_emb = nn.Embedding(n_ability, self.emb["ability"])
        self.item_emb    = nn.Embedding(n_item,    self.emb["item"])
        self.move_emb    = nn.Embedding(n_move,    self.emb["move"])

        self.use_move_attn = config["USE_MOVE_ATTENTION"]
        if self.use_move_attn:
            heads = config["MOVE_ATTENTION_HEADS"]
            self.move_attn_emb = nn.MultiheadAttention(self.emb["move"], heads, batch_first=True)
            self.move_attn_rdv = nn.MultiheadAttention(self.move_rdv_dim, heads, batch_first=True)

        self.input_dim = (self.emb["species"] + config["SPECIES_RDV_DIM"] + self.emb["ability"]
                          + self.emb["item"] + config["ITEM_RDV_DIM"]
                          + self.emb["move"] + config["MOVE_RDV_DIM"])
        self.mask_token = nn.Parameter(torch.randn(self.input_dim) * 0.02)
        self.input_proj = nn.Linear(self.input_dim, config["D_MODEL"])

        enc = nn.TransformerEncoderLayer(
            d_model=config["D_MODEL"], nhead=config["N_HEADS"],
            dim_feedforward=config["DIM_FEEDFORWARD"], dropout=config["DROPOUT"],
            batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=config["N_LAYERS"])

        self.head_species = nn.Linear(config["D_MODEL"], n_species)
        self.head_ability = nn.Linear(config["D_MODEL"], n_ability)
        self.head_item    = nn.Linear(config["D_MODEL"], n_item)
        self.head_moves   = nn.Linear(config["D_MODEL"], n_move)

        # Legality / team-construction buffers (values restored from checkpoint).
        self.register_buffer("ability_legal", torch.zeros(n_species, n_ability, dtype=torch.bool))
        self.register_buffer("move_legal",    torch.zeros(n_species, n_move,    dtype=torch.bool))
        self.register_buffer("is_mega",       torch.zeros(n_species, dtype=torch.bool))
        self.register_buffer("family_id",     torch.zeros(n_species, dtype=torch.long))
        self.register_buffer("form_to_stone_item", torch.full((n_species,), -1, dtype=torch.long))
        self.register_buffer("item_legal",    torch.ones(n_species, n_item, dtype=torch.bool))

    def _moveset(self, move_idx, move_rdv):
        B = move_idx.size(0)
        me = self.move_emb(move_idx)                       # [B,6,4,move_dim]
        if self.use_move_attn:
            e = me.reshape(B * 6, 4, self.emb["move"])
            e, _ = self.move_attn_emb(e, e, e)
            me = e.reshape(B, 6, 4, self.emb["move"])
            r = move_rdv.reshape(B * 6, 4, self.move_rdv_dim)
            r, _ = self.move_attn_rdv(r, r, r)
            move_rdv = r.reshape(B, 6, 4, self.move_rdv_dim)
        return me.mean(dim=2), move_rdv.mean(dim=2)

    def forward(self, batch):
        sp = self.species_emb(batch["species_idx"])
        ab = self.ability_emb(batch["ability_idx"])
        it = self.item_emb(batch["item_idx"])
        ms_emb, ms_rdv = self._moveset(batch["move_idx"], batch["move_rdv"])
        x = torch.cat([sp, batch["species_rdv"], ab,
                       it, batch["item_rdv"], ms_emb, ms_rdv], dim=-1)
        B = x.size(0)
        mpos = batch["mask_pos"]
        x = x.clone()
        x[torch.arange(B), mpos] = self.mask_token
        h = self.encoder(self.input_proj(x))
        hm = h[torch.arange(B), mpos]
        return {
            "species": self.head_species(hm),
            "ability": self.head_ability(hm),
            "item":    self.head_item(hm),
            "moves":   self.head_moves(hm),
        }

    @staticmethod
    def _filter_logits(logits, top_k, top_p):
        logits = logits.clone()
        if top_k and top_k >= 1:
            k = min(int(top_k), logits.size(-1))
            kth = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p and 0.0 < top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum = s_logits.softmax(-1).cumsum(-1)
            s_remove = cum > top_p
            s_remove[..., 1:] = s_remove[..., :-1].clone()
            s_remove[..., 0] = False
            remove = torch.zeros_like(s_remove).scatter(-1, s_idx, s_remove)
            logits = logits.masked_fill(remove, float("-inf"))
        return logits

    @staticmethod
    def _sampling_active(phase):
        if not ((TOP_K and TOP_K >= 1) or (TOP_P and 0.0 < TOP_P < 1.0)):
            return False
        return SAMPLING_PHASE == "both" or SAMPLING_PHASE == phase

    @torch.no_grad()
    def predict(self, batch, phase="predict", n_predictions=1, n_moves=4):
        """Decode the masked slot into legal, team-valid candidate Pokemon.

        Returns dict with species/ability/item [B,n] index tensors, moves (list[B]
        of list[n] of index tensors), and the raw `logits`. See the notebook for
        the full set of enforced constraints (legality, no dupes, Mega rules).
        """
        out = self.forward(batch)
        B, S = out["species"].shape
        active = self._sampling_active(phase) and n_predictions == 1
        sp_logits = out["species"].clone()

        pos  = torch.arange(6, device=sp_logits.device)
        keep = pos.unsqueeze(0) != batch["mask_pos"].unsqueeze(1)
        vis  = batch["species_idx"][keep].view(B, 5)
        vis_fam = self.family_id[vis]
        forbid  = (self.family_id.view(1, S, 1) == vis_fam.view(B, 1, 5)).any(-1)
        two_megas = self.is_mega[vis].sum(1) >= 2
        forbid = forbid | (two_megas.view(B, 1) & self.is_mega.view(1, S))
        sp_logits = sp_logits.masked_fill(forbid, float("-inf"))
        dead = torch.isinf(sp_logits).all(-1)
        if dead.any():
            sp_logits[dead] = out["species"][dead]

        if active:
            probs = F.softmax(self._filter_logits(sp_logits, TOP_K, TOP_P) / TEMP, -1)
            sp_cands = torch.multinomial(probs, 1)
        else:
            k = min(int(n_predictions), S)
            sp_cands = sp_logits.topk(k, dim=-1).indices
        n = sp_cands.size(1)

        ab_cands = torch.zeros_like(sp_cands)
        it_cands = torch.zeros_like(sp_cands)
        mv_cands = [[None] * n for _ in range(B)]

        for j in range(n):
            sp_j = sp_cands[:, j]
            ab_logits = out["ability"].clone()
            mv_logits = out["moves"].clone()
            it_logits = out["item"].clone()
            if ENFORCE_LEGALITY:
                ab_logits = ab_logits.masked_fill(~self.ability_legal[sp_j], float("-inf"))
                mv_logits = mv_logits.masked_fill(~self.move_legal[sp_j],    float("-inf"))
            it_logits = it_logits.masked_fill(~self.item_legal[sp_j], float("-inf"))

            if active:
                ab_cands[:, j] = torch.multinomial(ab_logits.softmax(-1), 1).squeeze(-1)
            else:
                ab_cands[:, j] = ab_logits.argmax(-1)

            it_j = it_logits.argmax(-1)
            forced = self.form_to_stone_item[sp_j]
            it_j = torch.where(forced >= 0, forced, it_j)
            it_cands[:, j] = it_j

            for i in range(B):
                legal = torch.isfinite(mv_logits[i])
                kk = int(min(n_moves, int(legal.sum().item()))) or 1
                if active:
                    mv_cands[i][j] = torch.multinomial(
                        mv_logits[i].softmax(-1), kk, replacement=False)
                else:
                    mv_cands[i][j] = mv_logits[i].topk(kk).indices

        return {"species": sp_cands, "ability": ab_cands, "item": it_cands,
                "moves": mv_cands, "logits": out}


# --------------------------------------------------------------------------- #
# Species name mapping: CNN Gen-I labels -> transformer species vocab
# --------------------------------------------------------------------------- #
def map_cnn_species_to_vocab(cnn_name, species_vocab):
    """Best-effort map a CNN display name to a key in `species_vocab`.

    Returns the matched vocab key, or None so the caller can ask the user to pick.
    """
    if cnn_name in species_vocab:
        return cnn_name
    regional = {"Alolan": "Alola", "Galarian": "Galar", "Hisuian": "Hisui", "Paldean": "Paldea"}
    parts = cnn_name.split(" ", 1)
    if len(parts) == 2 and parts[0] in regional:
        candidate = f"{parts[1]}-{regional[parts[0]]}"
        if candidate in species_vocab:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TeamPredictor:
    """Loads the checkpoint + RDV vectors and predicts a team's 6th Pokemon."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, rdv_dir=DEFAULT_RDV_DIR,
                 team_dir=DEFAULT_TEAM_DIR, device=None):
        checkpoint_path = Path(checkpoint_path)
        rdv_dir = Path(rdv_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        self.device = device or _pick_device()
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.vocabs = ckpt["vocabs"]            # name -> idx (with "<UNK>": 0)
        self.inv = {k: {i: n for n, i in v.items()} for k, v in self.vocabs.items()}

        self.rdvs = {}
        for key, fname in (("species", "pokemon_vectors.pkl"),
                           ("item", "item_vectors.pkl"),
                           ("move", "move_vectors.pkl")):
            path = rdv_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"RDV file not found: {path}")
            with open(path, "rb") as f:
                self.rdvs[key] = pickle.load(f)

        self._zero = {
            "species": [0.0] * self.config["SPECIES_RDV_DIM"],
            "item": [0.0] * self.config["ITEM_RDV_DIM"],
            "move": [0.0] * self.config["MOVE_RDV_DIM"],
        }

        self.model = MaskedTeamTransformer(
            self.config,
            n_species=len(self.vocabs["species"]),
            n_ability=len(self.vocabs["ability"]),
            n_item=len(self.vocabs["item"]),
            n_move=len(self.vocabs["move"]),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        # Per-species default sets (most-common ability/item/moveset) for UI
        # pre-fill. Optional: a missing/unreadable team dir just yields {}.
        self.defaults = self._build_defaults(team_dir)

    def _build_defaults(self, team_dir):
        """Aggregate the most-common ability/item/4-move set per species.

        Reads every `*.pkl` in `team_dir` (each a list of teams; each team six
        rows of `[species, ability, item, m1..m4]`). Names already match the
        vocab, so the result drops straight into the UI. Returns
        `{species: {ability, item, moves:[4]}}`, restricted to vocab species.
        """
        team_dir = Path(team_dir)
        if not team_dir.is_dir():
            return {}

        abilities = defaultdict(Counter)
        items = defaultdict(Counter)
        movesets = defaultdict(Counter)
        for path in sorted(team_dir.glob("*.pkl")):
            try:
                with open(path, "rb") as f:
                    teams = pickle.load(f)
            except Exception:
                continue
            for team in teams:
                for row in team:
                    if not row or not row[0]:
                        continue
                    species = row[0]
                    if len(row) > 1 and row[1]:
                        abilities[species][row[1]] += 1
                    if len(row) > 2 and row[2]:
                        items[species][row[2]] += 1
                    moves = tuple(row[3:7])
                    if any(moves):
                        movesets[species][moves] += 1

        defaults = {}
        species_vocab = self.vocabs["species"]
        for species in set(abilities) | set(items) | set(movesets):
            if species not in species_vocab:
                continue
            entry = {}
            if abilities[species]:
                entry["ability"] = abilities[species].most_common(1)[0][0]
            if items[species]:
                entry["item"] = items[species].most_common(1)[0][0]
            if movesets[species]:
                moves = list(movesets[species].most_common(1)[0][0])
                entry["moves"] = [m for m in moves if m]
            if entry:
                defaults[species] = entry
        return defaults

    def species_defaults(self):
        """{species: {ability, item, moves:[...]}} for UI pre-fill."""
        return self.defaults

    # -- vocab lists for the UI (drop the <UNK> sentinel) --------------------
    def vocab_lists(self):
        return {
            key: sorted(n for n in vocab if n != "<UNK>")
            for key, vocab in self.vocabs.items()
        }

    def _apply_mega(self, mon):
        """mon is [species, ability, item, m1..m4]; rename to Mega form if it holds a stone."""
        mega = MEGA_STONE_TO_FORM.get(mon[2])
        if mega is not None and mega in self.rdvs["species"]:
            mon = list(mon)
            mon[0] = mega
        return mon

    def _team_to_tensors(self, team):
        """team: list of 6 mons (each [species, ability, item, m1..m4])."""
        sp_idx, ab_idx, it_idx, mv_idx = [], [], [], []
        sp_rdv, it_rdv, mv_rdv = [], [], []
        for p in team:
            species, ability, item = p[0], p[1], p[2]
            moves = list(p[3:7]) + [""] * (4 - len(p[3:7]))
            sp_idx.append(self.vocabs["species"].get(species, 0))
            ab_idx.append(self.vocabs["ability"].get(ability, 0))
            it_idx.append(self.vocabs["item"].get(item, 0))
            mv_idx.append([self.vocabs["move"].get(m, 0) for m in moves])
            sp_rdv.append(self.rdvs["species"].get(species, self._zero["species"]))
            it_rdv.append(self.rdvs["item"].get(item, self._zero["item"]))
            mv_rdv.append([self.rdvs["move"].get(m, self._zero["move"]) for m in moves])
        dev = self.device
        return {
            "species_idx": torch.tensor([sp_idx], dtype=torch.long, device=dev),
            "ability_idx": torch.tensor([ab_idx], dtype=torch.long, device=dev),
            "item_idx":    torch.tensor([it_idx], dtype=torch.long, device=dev),
            "move_idx":    torch.tensor([mv_idx], dtype=torch.long, device=dev),
            "species_rdv": torch.tensor([sp_rdv], dtype=torch.float, device=dev),
            "item_rdv":    torch.tensor([it_rdv], dtype=torch.float, device=dev),
            "move_rdv":    torch.tensor([mv_rdv], dtype=torch.float, device=dev),
        }

    @staticmethod
    def _mon_from_dict(d):
        moves = (d.get("moves") or [])[:4]
        moves = moves + [""] * (4 - len(moves))
        return [d.get("species", ""), d.get("ability", ""), d.get("item", ""), *moves]

    @torch.no_grad()
    def predict_sixth(self, team5, n_candidates=5):
        """team5: list of up to 5 dicts {species, ability, item, moves:[...]}.

        Returns a list of `n_candidates` legal, team-valid 6th-Pokemon suggestions,
        best first. Each is {species, ability, item, moves:[...]}.
        """
        context = [self._apply_mega(self._mon_from_dict(d)) for d in team5[:5]]
        while len(context) < 5:
            context.append(["", "", "", "", "", "", ""])
        team = context + [["", "", "", "", "", "", ""]]      # slot 5 = masked target

        batch = self._team_to_tensors(team)
        batch["mask_pos"] = torch.tensor([5], dtype=torch.long, device=self.device)
        pred = self.model.predict(batch, phase="predict", n_predictions=n_candidates)

        inv = self.inv
        out = []
        for j in range(pred["species"].size(1)):
            out.append({
                "species": inv["species"][pred["species"][0, j].item()],
                "ability": inv["ability"][pred["ability"][0, j].item()],
                "item":    inv["item"][pred["item"][0, j].item()],
                "moves":   [inv["move"][m] for m in pred["moves"][0][j].tolist()],
            })
        return out
