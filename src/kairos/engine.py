"""Pool de engines Stockfish (UCI) com MultiPV, usado pela arvore de candidatos e pela interface."""
import queue

import chess
import chess.engine


class Analyzer:
    """Pool de engines; multipv() e bloqueante e thread-safe."""

    def __init__(self, path, n, hash_mb=16):

        self.q = queue.Queue()
        self.engines = []

        for _ in range(n):
            e = chess.engine.SimpleEngine.popen_uci(path)
            e.configure({"Threads": 1, "Hash": hash_mb})
            self.engines.append(e)
            self.q.put(e)

    def multipv(self, board, k, nodes, root_moves=None):
        """Lista ordenada [{move, cp, mate, pv}] do ponto de vista de quem joga."""

        e = self.q.get()

        try:
            infos = e.analyse(board, chess.engine.Limit(nodes=nodes),
                              multipv=k, root_moves=root_moves)
        finally:
            self.q.put(e)

        if isinstance(infos, dict):
            infos = [infos]

        out = []

        for d in infos:
            pv, sc = d.get("pv"), d.get("score")
            if not pv or sc is None:
                continue
            s = sc.pov(board.turn)
            out.append({"move": pv[0], "pv": list(pv),
                        "cp": s.score(mate_score=10000),
                        "mate": s.mate()})
        return out

    def close(self):
        for e in self.engines:
            try:
                e.quit()
            except Exception:
                pass
