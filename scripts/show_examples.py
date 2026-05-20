import csv

print('EXEMPLES DE CORRECTIONS SCIENTIFIQUES')
print('='*50)

with open('scientific_names_correction.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    examples = list(reader)

    # Afficher quelques exemples représentatifs
    categories_to_show = ['fungal', 'bacterial', 'pest', 'healthy']

    for category in categories_to_show:
        print(f'\n{category.upper()} EXAMPLES:')
        cat_examples = [ex for ex in examples if ex['category'] == category][:3]
        for ex in cat_examples:
            print(f'  "{ex["original_name"]}" -> "{ex["scientific_name"]}"')

    # Statistiques de confiance
    confidence_levels = {}
    for ex in examples:
        conf = ex['confidence']
        confidence_levels[conf] = confidence_levels.get(conf, 0) + 1

    print(f'\nNIVEAUX DE CONFIANCE:')
    total = len(examples)
    for level, count in confidence_levels.items():
        pct = count/total*100
        print(f'  {level}: {count} noms ({pct:.1f}%)')

    print(f'\nTotal de noms traités: {total}')