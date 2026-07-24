from pathlib import Path

folder = 'LatexConstants'

accuracyTableBegin = Path(f'{folder}/accuracyTableBegin.txt').read_text(encoding="utf-8")
accuracyTableEnd = Path(f'{folder}/accuracyTableEnd.txt').read_text(encoding="utf-8")

ablationAccuracyTableBegin = Path(f'{folder}/ablationAccuracyTableBegin.txt').read_text(encoding="utf-8")
ablationAccuracyTableEnd = Path(f'{folder}/ablationAccuracyTableEnd.txt').read_text(encoding="utf-8")

ciTableBegin = Path(f'{folder}/ciTableBegin.txt').read_text(encoding="utf-8")
ciTableEnd = Path(f'{folder}/ciTableEnd.txt').read_text(encoding="utf-8")

rankingTableBegin = Path(f'{folder}/rankingTableBegin.txt').read_text(encoding="utf-8")
rankingTableEnd = Path(f'{folder}/rankingTableEnd.txt').read_text(encoding="utf-8")

ablationCITableBegin = Path(f'{folder}/ablationCITableBegin.txt').read_text(encoding="utf-8")
ablationCITableEnd = Path(f'{folder}/ablationCITableEnd.txt').read_text(encoding="utf-8")

ablationRankingTableBegin = Path(f'{folder}/ablationRankingTableBegin.txt').read_text(encoding="utf-8")
ablationRankingTableEnd = Path(f'{folder}/ablationRankingTableEnd.txt').read_text(encoding="utf-8")