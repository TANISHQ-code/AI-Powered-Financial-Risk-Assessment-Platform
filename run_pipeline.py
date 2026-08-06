import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import compare_model_types


if __name__ == '__main__':
    results = compare_model_types()
    print('Model comparison results:')
    for model_type, metrics in results.items():
        print(f'\n{model_type}')
        for name, value in metrics.items():
            print(f'- {name}: {value}')

    out_file = Path(__file__).resolve().parent / 'model_comparison.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote model comparison results to {out_file}')
