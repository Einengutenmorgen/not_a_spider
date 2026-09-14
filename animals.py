"""Animal quiz data, restructured for automated answer checking.

Schema per entry:
    animal               canonical name
    accepted_answers     list[str]  - every name counted as correct
    description          str        - prose description (moved out of the feature list)
    unique_features      list[str]  - features used in this entry's questions that
                                      identify the animal on their own; each one
                                      matches exactly one entry in ANIMALS
    shared_features      list[list[str]]
                                    - one inner list per descriptor used in this
                                      entry's questions, decomposed into its
                                      features. Members need not be identifying
                                      on their own; the combination must be.
                                      A combination may repeat a unique feature -
                                      that is how the question was phrased.
    attributes           dict attribute -> answer spec, one of:
        {'type': 'set',   'display': str, 'accepted': list[str], 'note': str?}
        {'type': 'range', 'display': str, 'min': float, 'max': float,
         'unit': str, 'measure': str?}
    attribute_questions  unchanged
    jeopardy_questions   unchanged

Matching convention: compare through normalize() - lowercase, punctuation and
hyphens collapsed to spaces. A 'range' answer counts as correct if the first
number in the answer falls inside [min, max]; the unit is NOT checked (see
PARSING_NOTES.md).
"""

import json
import re
import sys


def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def check_answer(spec, answer):
    """True if `answer` satisfies an attribute spec."""
    if spec['type'] == 'set':
        return normalize(answer) in {normalize(a) for a in spec['accepted']}
    match = re.search(r'\d+(?:\.\d+)?', answer)
    return bool(match) and spec['min'] <= float(match.group()) <= spec['max']


def check_animal(entry, answer):
    """True if `answer` names the animal of `entry`."""
    return normalize(answer) in {normalize(a) for a in entry['accepted_answers']}


ANIMALS = [
    {
        'animal': 'Platypus',
        'accepted_answers': ['Platypus', 'Duck-billed platypus'],
        'description': 'A semi-aquatic Australian mammal that lays eggs, with a duck-like bill, webbed feet and venomous spurs.',
        'unique_features': [
            'duck-like bill',
            'venomous spurs',
        ],
        'shared_features': [
            ['mammal', 'semi-aquatic', 'native to Australia', 'lays eggs'],
            ['mammal', 'semi-aquatic', 'native to Australia', 'monotreme', 'lays eggs'],
            ['mammal', 'native to Australia', 'lays eggs', 'duck-like bill', 'webbed feet', 'venomous spurs'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Brown', 'accepted': ['brown']},
            'weight': {'type': 'range', 'display': '1 - 2.4 kg', 'min': 1.0, 'max': 2.4, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '10 - 17 years', 'min': 10, 'max': 17, 'unit': 'years'},
            'height': {'type': 'range', 'display': '40 - 50 cm', 'min': 40, 'max': 50, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the semi-aquatic Australian mammal that lays eggs have? Answer in one word.',
            'color': 'What is the color of the semi-aquatic Australian mammal that lays eggs? Answer in one word.',
            'weight': 'What is the weight of the semi-aquatic Australian mammal that lays eggs? Answer in one word.',
            'lifespan': 'What is the lifespan of the semi-aquatic Australian mammal that lays eggs? Answer in one word.',
            'height': 'What is the length of the semi-aquatic Australian mammal that lays eggs? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which semi-aquatic Australian mammal belongs to the monotreme order and lays eggs instead of giving birth?',
            'What egg-laying Australian mammal is famous for having a duck-like bill, webbed feet, and venomous spurs on the males?'
        ]
    },
    {
        'animal': 'Kangaroo',
        'accepted_answers': ['Kangaroo', 'Roo', 'Macropod'],
        'description': 'A large Australian marsupial that hops, using oversized hind legs and a muscular tail for locomotion.',
        'unique_features': [
        ],
        'shared_features': [
            ['marsupial', 'large', 'native to Australia', 'hops'],
            ['marsupial', 'large', 'native to Australia', 'hops', 'carries a joey in a pouch'],
            ['marsupial', 'native to Australia', 'hops', 'oversized hind legs', 'muscular tail used for locomotion'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Reddish-brown to grey', 'accepted': ['reddish-brown', 'red-brown', 'reddish', 'brown', 'grey', 'gray']},
            'weight': {'type': 'range', 'display': '35 - 90 kg', 'min': 35, 'max': 90, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '8 - 23 years', 'min': 8, 'max': 23, 'unit': 'years'},
            'height': {'type': 'range', 'display': '1.0 - 1.6 m', 'min': 1.0, 'max': 1.6, 'unit': 'm', 'measure': 'standing height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the large Australian marsupial that hops have? Answer in one word.',
            'color': 'What is the primary color of the large Australian marsupial that hops? Answer in one word.',
            'weight': 'What is the weight of the large Australian marsupial that hops? Answer in one word.',
            'lifespan': 'What is the lifespan of the large Australian marsupial that hops? Answer in one word.',
            'height': 'What is the height of the large Australian marsupial that hops? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which large Australian marsupial carries its joey in a pouch and moves by hopping?',
            'What Australian marsupial is known for its powerful hind legs, large muscular tail, and hopping locomotion?'
        ]
    },
    {
        'animal': 'Chameleon',
        'accepted_answers': ['Chameleon', 'Chamaeleonidae'],
        'description': 'A lizard that can rapidly change its skin color, with independently mobile cone-shaped eyes and a prehensile tail.',
        'unique_features': [
            'rapidly changes skin color',
        ],
        'shared_features': [
            ['lizard', 'rapidly changes skin color'],
            ['lizard', 'rapidly changes skin color', 'sticky extrudable tongue'],
            ['lizard', 'rapidly changes skin color', 'prehensile tail', 'cone-shaped eyes', 'independently mobile eyes'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Variable (Green, brown, yellow)', 'accepted': ['variable', 'green', 'brown', 'yellow']},
            'weight': {'type': 'range', 'display': '0.01 - 0.2 kg', 'min': 0.01, 'max': 0.2, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '2 - 10 years', 'min': 2, 'max': 10, 'unit': 'years'},
            'height': {'type': 'range', 'display': '15 - 60 cm', 'min': 15, 'max': 60, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the lizard that can rapidly change its skin color have? Answer in one word.',
            'color': 'What is the color of the lizard that can rapidly change its skin color? Answer in one word.',
            'weight': 'What is the weight of the lizard that can rapidly change its skin color? Answer in one word.',
            'lifespan': 'What is the lifespan of the lizard that can rapidly change its skin color? Answer in one word.',
            'height': 'What is the length of the lizard that can rapidly change its skin color? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which lizard is famous for changing its skin color for camouflage and having a rapidly extrudable sticky tongue?',
            'What specific color-changing lizard has a curled prehensile tail and cone-shaped eyes that can rotate independently?'
        ]
    },
    {
        'animal': 'Narwhal',
        'accepted_answers': ['Narwhal', 'Narwhale', 'Monodon monoceros'],
        'description': 'An Arctic whale with a single long spiral tusk, colloquially nicknamed the unicorn of the sea.',
        'unique_features': [
            'single long spiral tusk',
            'nicknamed unicorn of the sea',
        ],
        'shared_features': [
            ['whale', 'Arctic', 'single long spiral tusk'],
            ['whale', 'Arctic', 'single long spiral tusk', 'nicknamed unicorn of the sea'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '2 (flippers)', 'accepted': ['2', 'two'], 'note': 'flippers, not legs'},
            'color': {'type': 'set', 'display': 'Mottled grey-black and white', 'accepted': ['mottled', 'grey', 'gray', 'grey-black', 'black', 'white']},
            'weight': {'type': 'range', 'display': '800 - 1600 kg', 'min': 800, 'max': 1600, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '50 - 55 years', 'min': 50, 'max': 55, 'unit': 'years'},
            'height': {'type': 'range', 'display': '3.95 - 5.5 m', 'min': 3.95, 'max': 5.5, 'unit': 'm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many flippers does the Arctic whale with a single long spiral tusk have? Answer in one word.',
            'color': 'What is the color of the Arctic whale with a single long spiral tusk? Answer in one word.',
            'weight': 'What is the weight of the Arctic whale with a single long spiral tusk? Answer in one word.',
            'lifespan': 'What is the lifespan of the Arctic whale with a single long spiral tusk? Answer in one word.',
            'height': 'What is the length of the Arctic whale with a single long spiral tusk? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which Arctic whale species features males with a single, long, spiral tusk protruding from their heads?',
            'What medium-sized toothed whale from the Arctic is widely nicknamed the "unicorn of the sea" due to its long tusk?'
        ]
    },
    {
        'animal': 'Sloth',
        'accepted_answers': ['Sloth', 'Three-toed sloth', 'Two-toed sloth', 'Tree sloth'],
        'description': 'An incredibly slow-moving arboreal mammal from the Americas that spends most of its life hanging upside down and grows algae in its fur.',
        'unique_features': [
            'grows algae in its fur',
        ],
        'shared_features': [
            ['mammal', 'arboreal', 'slow-moving', 'hangs upside down'],
            ['mammal', 'arboreal', 'slow-moving', 'native to the Americas', 'grows algae in its fur'],
            ['mammal', 'arboreal', 'native to the Americas', 'hangs upside down'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Brownish-grey to greenish', 'accepted': ['brownish-grey', 'brown', 'grey', 'gray', 'greenish', 'green']},
            'weight': {'type': 'range', 'display': '3.5 - 4.5 kg', 'min': 3.5, 'max': 4.5, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '20 - 30 years', 'min': 20, 'max': 30, 'unit': 'years'},
            'height': {'type': 'range', 'display': '45 - 60 cm', 'min': 45, 'max': 60, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the incredibly slow-moving arboreal mammal that hangs upside down have? Answer in one word.',
            'color': 'What is the color of the incredibly slow-moving arboreal mammal that hangs upside down? Answer in one word.',
            'weight': 'What is the weight of the incredibly slow-moving arboreal mammal that hangs upside down? Answer in one word.',
            'lifespan': 'What is the lifespan of the incredibly slow-moving arboreal mammal that hangs upside down? Answer in one word.',
            'height': 'What is the length of the incredibly slow-moving arboreal mammal that hangs upside down? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which incredibly slow-moving Central and South American tree-dwelling mammal grows algae in its fur?',
            'What arboreal mammal from the Americas spends most of its life hanging upside down and comes in three-toed or two-toed varieties?'
        ]
    },
    {
        'animal': 'Elephant',
        'accepted_answers': ['Elephant', 'African elephant', 'Asian elephant'],
        'description': 'The largest living land mammal, with large floppy ears and a long muscular trunk used for grasping and drinking.',
        'unique_features': [
            'largest living land mammal',
            'long muscular trunk',
        ],
        'shared_features': [
            ['mammal', 'largest living land mammal', 'long muscular trunk'],
            ['mammal', 'large', 'large floppy ears', 'tusks', 'long muscular trunk'],
            ['largest living land mammal', 'trunk used for grasping and drinking'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Grey', 'accepted': ['grey', 'gray']},
            'weight': {'type': 'range', 'display': '3000 - 6000 kg', 'min': 3000, 'max': 6000, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '60 - 70 years', 'min': 60, 'max': 70, 'unit': 'years'},
            'height': {'type': 'range', 'display': '2.5 - 4.0 m', 'min': 2.5, 'max': 4.0, 'unit': 'm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the largest living land mammal with a long muscular trunk have? Answer in one word.',
            'color': 'What is the color of the largest living land mammal with a long muscular trunk? Answer in one word.',
            'weight': 'What is the weight of the largest living land mammal with a long muscular trunk? Answer in one word.',
            'lifespan': 'What is the lifespan of the largest living land mammal with a long muscular trunk? Answer in one word.',
            'height': 'What is the height of the largest living land mammal with a long muscular trunk? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which massive terrestrial mammal has large floppy ears, ivory tusks, and a long muscular prehensile trunk?',
            'What is the largest living land animal, known for using its long trunk for breathing, drinking, and grasping?'
        ]
    },
    {
        'animal': 'Axolotl',
        'accepted_answers': ['Axolotl', 'Axolote', 'Mexican walking fish', 'Ambystoma mexicanum'],
        'description': 'A Mexican salamander that retains feathery external gills into adulthood; an aquatic amphibian natively found only in Lake Xochimilco.',
        'unique_features': [
            'found only in Lake Xochimilco',
            'retains feathery external gills into adulthood',
        ],
        'shared_features': [
            ['salamander', 'aquatic', 'native to Mexico', 'retains feathery external gills into adulthood'],
            ['salamander', 'native to Mexico', 'found only in Lake Xochimilco', 'retains feathery external gills into adulthood'],
            ['amphibian', 'native to Mexico', 'neotenic', 'found only in Lake Xochimilco', 'regenerates lost limbs'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Pinkish, brown, or black', 'accepted': ['pinkish', 'pink', 'brown', 'black']},
            'weight': {'type': 'range', 'display': '0.1 - 0.3 kg', 'min': 0.1, 'max': 0.3, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '10 - 15 years', 'min': 10, 'max': 15, 'unit': 'years'},
            'height': {'type': 'range', 'display': '15 - 45 cm', 'min': 15, 'max': 45, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the aquatic Mexican salamander that retains external gills into adulthood have? Answer in one word.',
            'color': 'What is the color of the aquatic Mexican salamander that retains external gills into adulthood? Answer in one word.',
            'weight': 'What is the weight of the aquatic Mexican salamander that retains external gills into adulthood? Answer in one word.',
            'lifespan': 'What is the lifespan of the aquatic Mexican salamander that retains external gills into adulthood? Answer in one word.',
            'height': 'What is the length of the aquatic Mexican salamander that retains external gills into adulthood? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which critically endangered Mexican salamander, often mistakenly called a "walking fish", famously lives in Lake Xochimilco and retains feathery external gills into adulthood?',
            'What specific neotenic Mexican amphibian from Lake Xochimilco is heavily studied in science for its unique ability to regenerate entire lost limbs?'
        ]
    },
    {
        'animal': 'Giraffe',
        'accepted_answers': ['Giraffe'],
        'description': 'The tallest living terrestrial animal in the world, which uses a long prehensile blue-black tongue to eat acacia leaves.',
        'unique_features': [
            'tallest living terrestrial animal',
            'exceptionally long neck',
            'prehensile blue-black tongue',
        ],
        'shared_features': [
            ['ungulate', 'native to Africa', 'tallest living terrestrial animal', 'exceptionally long neck', 'spotted coat'],
            ['mammal', 'native to Africa', 'exceptionally long neck', 'prehensile blue-black tongue', 'eats acacia leaves'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Tan with brown irregular patches', 'accepted': ['tan', 'brown', 'tan with brown patches', 'spotted', 'patched']},
            'weight': {'type': 'range', 'display': '800 - 1200 kg', 'min': 800, 'max': 1200, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '20 - 25 years', 'min': 20, 'max': 25, 'unit': 'years'},
            'height': {'type': 'range', 'display': '4.3 - 5.7 m', 'min': 4.3, 'max': 5.7, 'unit': 'm', 'measure': 'total height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the tallest living terrestrial animal in the world have? Answer in one word.',
            'color': 'What is the coat color of the tallest living terrestrial animal in the world? Answer in one word.',
            'weight': 'What is the weight of the tallest living terrestrial animal in the world? Answer in one word.',
            'lifespan': 'What is the lifespan of the tallest living terrestrial animal in the world? Answer in one word.',
            'height': 'What is the height of the tallest living terrestrial animal in the world? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which African even-toed ungulate is the tallest living terrestrial animal and is famous for its exceptionally long neck and spotted coat?',
            'What incredibly tall African mammal has the longest neck of any living animal and uses a prehensile blue-black tongue to eat thorny acacia leaves?'
        ]
    },
    {
        'animal': 'Pangolin',
        'accepted_answers': ['Pangolin', 'Scaly anteater', 'Trenggiling'],
        'description': 'The only mammal entirely covered in overlapping keratin scales; it rolls into a tight protective ball when threatened.',
        'unique_features': [
            'only mammal entirely covered in overlapping keratin scales',
        ],
        'shared_features': [
            ['mammal', 'nocturnal', 'native to Africa and Asia', 'only mammal entirely covered in overlapping keratin scales', 'rolls into a protective ball when threatened'],
            ['mammal', 'only mammal entirely covered in overlapping keratin scales', 'long sticky tongue', 'eats ants'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Brown to olive-yellow', 'accepted': ['brown', 'olive-yellow', 'olive', 'yellow']},
            'weight': {'type': 'range', 'display': '2 - 35 kg', 'min': 2, 'max': 35, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '20 - 22 years', 'min': 20, 'max': 22, 'unit': 'years'},
            'height': {'type': 'range', 'display': '30 - 100 cm', 'min': 30, 'max': 100, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the only mammal entirely covered in overlapping keratin scales have? Answer in one word.',
            'color': 'What is the color of the only mammal entirely covered in overlapping keratin scales? Answer in one word.',
            'weight': 'What is the weight of the only mammal entirely covered in overlapping keratin scales? Answer in one word.',
            'lifespan': 'What is the lifespan of the only mammal entirely covered in overlapping keratin scales? Answer in one word.',
            'height': 'What is the length of the only mammal entirely covered in overlapping keratin scales? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which primarily nocturnal mammal from Africa and Asia is covered entirely in overlapping pinecone-like keratin scales and curls into a ball?',
            'Often called a scaly anteater, what heavily trafficked mammal is the only mammal fully covered in keratin scales and catches ants with a long sticky tongue?'
        ]
    },
    {
        'animal': 'Walrus',
        'accepted_answers': ['Walrus', 'Odobenus rosmarus'],
        'description': 'A large Arctic marine mammal with prominent tusks and thick whiskers, using highly sensitive vibrissae to forage for clams on the ocean floor.',
        'unique_features': [
            'thick bristly whiskers',
            'forages for clams on the ocean floor',
        ],
        'shared_features': [
            ['mammal', 'marine', 'Arctic', 'large', 'tusks', 'thick bristly whiskers'],
            ['mammal', 'marine', 'Arctic', 'large', 'flippers', 'tusks', 'thick bristly whiskers'],
            ['pinniped', 'Arctic', 'tusks', 'thick bristly whiskers', 'forages for clams on the ocean floor'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4 (flippers)', 'accepted': ['4', 'four'], 'note': 'flippers, not legs'},
            'color': {'type': 'set', 'display': 'Cinnamon-brown', 'accepted': ['cinnamon-brown', 'cinnamon', 'brown']},
            'weight': {'type': 'range', 'display': '800 - 1700 kg', 'min': 800, 'max': 1700, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '20 - 30 years', 'min': 20, 'max': 30, 'unit': 'years'},
            'height': {'type': 'range', 'display': '2.2 - 3.6 m', 'min': 2.2, 'max': 3.6, 'unit': 'm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many flippers does the large Arctic marine mammal with prominent tusks and thick whiskers have? Answer in one word.',
            'color': 'What is the color of the large Arctic marine mammal with prominent tusks and thick whiskers? Answer in one word.',
            'weight': 'What is the weight of the large Arctic marine mammal with prominent tusks and thick whiskers? Answer in one word.',
            'lifespan': 'What is the lifespan of the large Arctic marine mammal with prominent tusks and thick whiskers? Answer in one word.',
            'height': 'What is the length of the large Arctic marine mammal with prominent tusks and thick whiskers? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which large flippered Arctic marine mammal is instantly recognized by its prominent downward-pointing tusks and thick bristly whiskers?',
            'What massive Arctic pinniped has two very long canine tusks and uses its highly sensitive, bristly whiskers to forage for clams?'
        ]
    },
    {
        'animal': 'Flamingo',
        'accepted_answers': ['Flamingo', 'Phoenicopteridae'],
        'description': 'A wading bird that gets its pink plumage from beta-carotene in its diet and filter-feeds with its head upside down in shallow muddy water.',
        'unique_features': [
            'pink plumage from beta-carotene',
            'filter-feeds with its head upside down',
        ],
        'shared_features': [
            ['bird', 'wading bird', 'pink plumage from beta-carotene'],
            ['bird', 'long legs', 'pink plumage from beta-carotene', 'filter-feeds with its head upside down', 'bent bill'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4 (2 legs, 2 wings)', 'accepted': ['4', 'four'], 'note': '2 legs and 2 wings'},
            'color': {'type': 'set', 'display': 'Pink', 'accepted': ['pink']},
            'weight': {'type': 'range', 'display': '2 - 4 kg', 'min': 2, 'max': 4, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '20 - 30 years', 'min': 20, 'max': 30, 'unit': 'years'},
            'height': {'type': 'range', 'display': '1.1 - 1.5 m', 'min': 1.1, 'max': 1.5, 'unit': 'm', 'measure': 'standing height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs in total (including wings) does the wading bird that gets its plumage from beta-carotene have? Answer in one word.',
            'color': 'What is the primary color of the wading bird that gets its plumage from beta-carotene? Answer in one word.',
            'weight': 'What is the weight of the wading bird that gets its plumage from beta-carotene? Answer in one word.',
            'lifespan': 'What is the lifespan of the wading bird that gets its plumage from beta-carotene? Answer in one word.',
            'height': 'What is the height of the wading bird that gets its plumage from beta-carotene? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which tall wading bird gets its distinctive pink coloration from the beta-carotene in its diet of brine shrimp and algae?',
            'What long-legged pink bird feeds by turning its head upside down to filter out food from muddy water using a specialized bent bill?'
        ]
    },
    {
        'animal': 'Ostrich',
        'accepted_answers': ['Ostrich', 'Common ostrich'],
        'description': 'A flightless African bird and the heaviest living bird species; a fast-running bird with only two toes on each foot.',
        'unique_features': [
            'heaviest living bird species',
            'only two toes on each foot',
        ],
        'shared_features': [
            ['bird', 'flightless', 'heaviest living bird species'],
            ['bird', 'flightless', 'native to Africa', 'heaviest living bird species'],
            ['bird', 'flightless', 'fast-running', 'only two toes on each foot'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4 (2 legs, 2 wings)', 'accepted': ['4', 'four'], 'note': '2 legs and 2 wings'},
            'color': {'type': 'set', 'display': 'Black and white (males), brown-grey (females)', 'accepted': ['black and white', 'black', 'white', 'brown-grey', 'brown', 'grey', 'gray']},
            'weight': {'type': 'range', 'display': '63 - 145 kg', 'min': 63, 'max': 145, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '30 - 40 years', 'min': 30, 'max': 40, 'unit': 'years'},
            'height': {'type': 'range', 'display': '2.1 - 2.8 m', 'min': 2.1, 'max': 2.8, 'unit': 'm', 'measure': 'standing height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs in total does the heaviest living flightless bird species have? Answer in one word.',
            'color': 'What is the feather color of the heaviest living flightless bird species? Answer in one word.',
            'weight': 'What is the weight of the heaviest living flightless bird species? Answer in one word.',
            'lifespan': 'What is the lifespan of the heaviest living flightless bird species? Answer in one word.',
            'height': 'What is the height of the heaviest living flightless bird species? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which flightless African bird is the largest and heaviest living bird species in the world?',
            'What fast-running flightless bird is unique among all bird species for having only two toes on each foot?'
        ]
    },
    {
        'animal': 'Bat',
        'accepted_answers': ['Bat', 'Bats', 'Chiroptera'],
        'description': 'The only mammal naturally capable of true and sustained powered flight; it uses echolocation to navigate in darkness.',
        'unique_features': [
            'only mammal capable of true sustained powered flight',
        ],
        'shared_features': [
            ['mammal', 'only mammal capable of true sustained powered flight', 'uses echolocation', 'nocturnal', 'hunts insects in the dark'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4 (2 legs, 2 wings)', 'accepted': ['4', 'four'], 'note': '2 legs and 2 wings'},
            'color': {'type': 'set', 'display': 'Black, brown, or grey', 'accepted': ['black', 'brown', 'grey', 'gray']},
            'weight': {'type': 'range', 'display': '0.002 - 1.2 kg', 'min': 0.002, 'max': 1.2, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '10 - 30 years', 'min': 10, 'max': 30, 'unit': 'years'},
            'height': {'type': 'range', 'display': '3 - 40 cm', 'min': 3, 'max': 40, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the only mammal naturally capable of sustained powered flight have? Answer in one word.',
            'color': 'What is the typical color of the only mammal naturally capable of sustained powered flight? Answer in one word.',
            'weight': 'What is the weight of the only mammal naturally capable of sustained powered flight? Answer in one word.',
            'lifespan': 'What is the lifespan of the only mammal naturally capable of sustained powered flight? Answer in one word.',
            'height': 'What is the length of the only mammal naturally capable of sustained powered flight? Answer in one word.'
        },
        'jeopardy_questions': [
            'What is the only mammal species naturally capable of true and sustained powered flight?',
            'Which flying mammal famously uses echolocation to navigate and hunt insects in the dark?'
        ]
    },
    {
        'animal': 'Nautilus',
        'accepted_answers': ['Nautilus', 'Chambered nautilus'],
        'description': 'A marine cephalopod with a smooth, chambered external spiral shell, controlling its buoyancy by pumping fluid and gas into the shell chambers.',
        'unique_features': [
            'smooth chambered external spiral shell',
            'controls buoyancy with fluid and gas in its shell chambers',
        ],
        'shared_features': [
            ['marine', 'cephalopod', 'smooth chambered external spiral shell'],
            ['marine', 'cephalopod', 'smooth chambered external spiral shell', 'living fossil'],
            ['marine', 'cephalopod', 'about ninety tentacles', 'controls buoyancy with fluid and gas in its shell chambers'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '90 (tentacles)', 'accepted': ['90', 'ninety'], 'note': 'tentacles, not limbs'},
            'color': {'type': 'set', 'display': 'White with brown or red stripes', 'accepted': ['white', 'brown', 'red', 'striped']},
            'weight': {'type': 'range', 'display': '1.3 - 1.7 kg', 'min': 1.3, 'max': 1.7, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '15 - 20 years', 'min': 15, 'max': 20, 'unit': 'years'},
            'height': {'type': 'range', 'display': '20 - 25 cm', 'min': 20, 'max': 25, 'unit': 'cm', 'measure': 'shell diameter'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many tentacles does the marine cephalopod with a smooth, chambered external spiral shell have? Answer in one word.',
            'color': 'What is the shell color of the marine cephalopod with a smooth, chambered external spiral shell? Answer in one word.',
            'weight': 'What is the weight of the marine cephalopod with a smooth, chambered external spiral shell? Answer in one word.',
            'lifespan': 'What is the lifespan of the marine cephalopod with a smooth, chambered external spiral shell? Answer in one word.',
            'height': 'What is the diameter of the marine cephalopod with a smooth, chambered external spiral shell? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which marine cephalopod is considered a living fossil and has a smooth, chambered external spiral shell?',
            'What tentacled marine mollusk controls its buoyancy by pumping fluid and gas into the chambers of its external spiral shell?'
        ]
    },
    {
        'animal': 'Seahorse',
        'accepted_answers': ['Seahorse', 'Sea horse', 'Hippocampus'],
        'description': 'A small upright-swimming marine fish with a prehensile tail, famous for males that get pregnant and give birth.',
        'unique_features': [
            'males get pregnant and give birth',
            'horse-shaped head',
        ],
        'shared_features': [
            ['fish', 'marine', 'swims upright', 'prehensile tail'],
            ['fish', 'marine', 'prehensile tail', 'males get pregnant and give birth'],
            ['fish', 'marine', 'swims upright', 'horse-shaped head'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '0 (has fins instead)', 'accepted': ['0', 'zero', 'none'], 'note': 'has fins instead of limbs'},
            'color': {'type': 'set', 'display': 'Variable (Yellow, green, brown)', 'accepted': ['variable', 'yellow', 'green', 'brown']},
            'weight': {'type': 'range', 'display': '0.01 - 0.3 kg', 'min': 0.01, 'max': 0.3, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '1 - 5 years', 'min': 1, 'max': 5, 'unit': 'years'},
            'height': {'type': 'range', 'display': '1.5 - 35 cm', 'min': 1.5, 'max': 35, 'unit': 'cm', 'measure': 'upright height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the small upright-swimming marine fish with a prehensile tail have? Answer in one word.',
            'color': 'What is the color of the small upright-swimming marine fish with a prehensile tail? Answer in one word.',
            'weight': 'What is the weight of the small upright-swimming marine fish with a prehensile tail? Answer in one word.',
            'lifespan': 'What is the lifespan of the small upright-swimming marine fish with a prehensile tail? Answer in one word.',
            'height': 'What is the height of the small upright-swimming marine fish with a prehensile tail? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which small marine fish has a curled prehensile tail and is famous for males that get pregnant and give birth?',
            'What upright-swimming marine fish has a head that heavily resembles a tiny horse?'
        ]
    },
    {
        'animal': 'Giant Panda',
        'accepted_answers': ['Giant Panda', 'Panda', 'Panda bear'],
        'description': 'A large black-and-white bear native to China that feeds almost exclusively on bamboo.',
        'unique_features': [
            'feeds almost exclusively on bamboo',
        ],
        'shared_features': [
            ['bear', 'feeds almost exclusively on bamboo'],
            ['bear', 'large', 'black-and-white coloration', 'native to China', 'feeds almost exclusively on bamboo'],
            ['bear', 'large', 'black patches around the eyes', 'feeds almost exclusively on bamboo'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Black and white', 'accepted': ['black and white', 'black', 'white']},
            'weight': {'type': 'range', 'display': '70 - 120 kg', 'min': 70, 'max': 120, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '20 - 30 years', 'min': 20, 'max': 30, 'unit': 'years'},
            'height': {'type': 'range', 'display': '60 - 90 cm', 'min': 60, 'max': 90, 'unit': 'cm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the bear that feeds almost exclusively on bamboo have? Answer in one word.',
            'color': 'What is the color of the bear that feeds almost exclusively on bamboo? Answer in one word.',
            'weight': 'What is the weight of the bear that feeds almost exclusively on bamboo? Answer in one word.',
            'lifespan': 'What is the lifespan of the bear that feeds almost exclusively on bamboo? Answer in one word.',
            'height': 'What is the height of the bear that feeds almost exclusively on bamboo? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which large black-and-white bear native to China feeds almost exclusively on bamboo?',
            'What large bear species is famous for the distinct black patches around its eyes, ears, and across its round body while eating bamboo?'
        ]
    },
    {
        'animal': 'Hippopotamus',
        'accepted_answers': ['Hippopotamus', 'Hippo', 'Common hippopotamus'],
        'description': 'A barrel-shaped African river mammal that secretes a red blood sweat; a highly aggressive semi-aquatic mammal with enormous jaws and canine tusks.',
        'unique_features': [
            'secretes red blood sweat',
        ],
        'shared_features': [
            ['mammal', 'semi-aquatic', 'native to Africa', 'barrel-shaped body', 'secretes red blood sweat'],
            ['mammal', 'semi-aquatic', 'native to Africa', 'highly aggressive', 'barrel-shaped body', 'very short legs', 'enormous jaws', 'tusks'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Greyish-purple to brown', 'accepted': ['greyish-purple', 'grey', 'gray', 'purple', 'brown']},
            'weight': {'type': 'range', 'display': '1300 - 3200 kg', 'min': 1300, 'max': 3200, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '40 - 50 years', 'min': 40, 'max': 50, 'unit': 'years'},
            'height': {'type': 'range', 'display': '1.3 - 1.6 m', 'min': 1.3, 'max': 1.6, 'unit': 'm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the barrel-shaped African river mammal that secretes a red blood sweat have? Answer in one word.',
            'color': 'What is the color of the barrel-shaped African river mammal that secretes a red blood sweat? Answer in one word.',
            'weight': 'What is the weight of the barrel-shaped African river mammal that secretes a red blood sweat? Answer in one word.',
            'lifespan': 'What is the lifespan of the barrel-shaped African river mammal that secretes a red blood sweat? Answer in one word.',
            'height': 'What is the height of the barrel-shaped African river mammal that secretes a red blood sweat? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which massive, semi-aquatic African mammal secretes a reddish natural sunscreen fluid often called "blood sweat"?',
            'What highly aggressive, barrel-shaped African river mammal has very short legs, enormous jaws, and large canine tusks?'
        ]
    },
    {
        'animal': 'Toucan',
        'accepted_answers': ['Toucan', 'Toco toucan'],
        'description': 'A Neotropical bird with a giant, brightly colored beak, disproportionately large and supported inside by a lightweight bony honeycomb.',
        'unique_features': [
            'giant brightly colored beak',
            'lightweight bony honeycomb inside the bill',
        ],
        'shared_features': [
            ['bird', 'Neotropical', 'giant brightly colored beak'],
            ['bird', 'Neotropical', 'disproportionately large bill', 'lightweight bony honeycomb inside the bill'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4 (2 legs, 2 wings)', 'accepted': ['4', 'four'], 'note': '2 legs and 2 wings'},
            'color': {'type': 'set', 'display': 'Black with brightly colored bill and accents', 'accepted': ['black', 'multicolored', 'colorful']},
            'weight': {'type': 'range', 'display': '0.13 - 0.68 kg', 'min': 0.13, 'max': 0.68, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '15 - 20 years', 'min': 15, 'max': 20, 'unit': 'years'},
            'height': {'type': 'range', 'display': '29 - 63 cm', 'min': 29, 'max': 63, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs in total does the Neotropical bird with a giant, brightly colored beak have? Answer in one word.',
            'color': 'What is the primary color of the Neotropical bird with a giant, brightly colored beak? Answer in one word.',
            'weight': 'What is the weight of the Neotropical bird with a giant, brightly colored beak? Answer in one word.',
            'lifespan': 'What is the lifespan of the Neotropical bird with a giant, brightly colored beak? Answer in one word.',
            'height': 'What is the length of the Neotropical bird with a giant, brightly colored beak? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which tropical South American bird is famous for its enormous, brightly colored, yet surprisingly lightweight beak?',
            'What Neotropical bird possesses a giant, brightly colored bill supported inside by a lightweight bony honeycomb structure?'
        ]
    },
    {
        'animal': 'Skunk',
        'accepted_answers': ['Skunk', 'Skunks', 'Mephitidae'],
        'description': 'A black-and-white striped mammal famous for its foul-smelling spray, raising its bushy tail to warn predators before spraying from its anal glands.',
        'unique_features': [
            'sprays a foul-smelling liquid',
        ],
        'shared_features': [
            ['mammal', 'sprays a foul-smelling liquid'],
            ['mammal', 'native to the Americas', 'black-and-white coloration', 'striped', 'sprays a foul-smelling liquid'],
            ['mammal', 'raises its bushy tail as a warning', 'anal scent glands', 'sprays a foul-smelling liquid'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Black and white stripes', 'accepted': ['black and white', 'black', 'white', 'striped']},
            'weight': {'type': 'range', 'display': '1.5 - 6 kg', 'min': 1.5, 'max': 6, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '7 - 10 years', 'min': 7, 'max': 10, 'unit': 'years'},
            'height': {'type': 'range', 'display': '15 - 25 cm', 'min': 15, 'max': 25, 'unit': 'cm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the mammal famous for its foul-smelling spray have? Answer in one word.',
            'color': 'What is the color of the mammal famous for its foul-smelling spray? Answer in one word.',
            'weight': 'What is the weight of the mammal famous for its foul-smelling spray? Answer in one word.',
            'lifespan': 'What is the lifespan of the mammal famous for its foul-smelling spray? Answer in one word.',
            'height': 'What is the height of the mammal famous for its foul-smelling spray? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which black-and-white striped North American mammal is famous for spraying a foul-smelling liquid as a defense mechanism?',
            'What small mammalian omnivore raises its bushy tail to warn predators before spraying a highly pungent chemical from its anal glands?'
        ]
    },
    {
        'animal': 'Camel',
        'accepted_answers': ['Camel', 'Dromedary', 'Bactrian camel'],
        'description': 'A large desert ungulate with prominent fatty humps on its back, with thick eyelashes and sealable nostrils to survive harsh sandstorms.',
        'unique_features': [
            'fatty humps on its back',
        ],
        'shared_features': [
            ['ungulate', 'large', 'desert-dwelling', 'fatty humps on its back'],
            ['ungulate', 'large', 'desert-dwelling', 'beast of burden', 'fatty humps on its back'],
            ['ungulate', 'desert-dwelling', 'thick eyelashes', 'sealable nostrils', 'fatty humps on its back'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Light brown to tan', 'accepted': ['light brown', 'brown', 'tan', 'beige']},
            'weight': {'type': 'range', 'display': '300 - 1000 kg', 'min': 300, 'max': 1000, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '40 - 50 years', 'min': 40, 'max': 50, 'unit': 'years'},
            'height': {'type': 'range', 'display': '1.8 - 2.0 m', 'min': 1.8, 'max': 2.0, 'unit': 'm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the large desert ungulate with prominent fatty humps on its back have? Answer in one word.',
            'color': 'What is the color of the large desert ungulate with prominent fatty humps on its back? Answer in one word.',
            'weight': 'What is the weight of the large desert ungulate with prominent fatty humps on its back? Answer in one word.',
            'lifespan': 'What is the lifespan of the large desert ungulate with prominent fatty humps on its back? Answer in one word.',
            'height': 'What is the height of the large desert ungulate with prominent fatty humps on its back? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which large, long-legged desert beast of burden is famous for having fatty humps on its back?',
            'What large desert ungulate has thick eyelashes, sealable nostrils, and humps on its back to survive harsh, dry environments?'
        ]
    },
    {
        'animal': 'Cheetah',
        'accepted_answers': ['Cheetah', 'Acinonyx jubatus'],
        'description': 'A large spotted African cat famous for being the fastest land animal, with distinctive black tear-like streaks running down its face.',
        'unique_features': [
            'fastest land animal',
            'black tear-like facial streaks',
        ],
        'shared_features': [
            ['wild cat', 'large', 'native to Africa', 'spotted coat', 'fastest land animal'],
            ['wild cat', 'spotted coat', 'black tear-like facial streaks', 'fastest land animal'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Tan with black spots', 'accepted': ['tan', 'black', 'spotted', 'tan with black spots', 'yellow']},
            'weight': {'type': 'range', 'display': '21 - 72 kg', 'min': 21, 'max': 72, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '10 - 12 years', 'min': 10, 'max': 12, 'unit': 'years'},
            'height': {'type': 'range', 'display': '70 - 90 cm', 'min': 70, 'max': 90, 'unit': 'cm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the large spotted African cat famous for being the fastest land animal have? Answer in one word.',
            'color': 'What is the color of the large spotted African cat famous for being the fastest land animal? Answer in one word.',
            'weight': 'What is the weight of the large spotted African cat famous for being the fastest land animal? Answer in one word.',
            'lifespan': 'What is the lifespan of the large spotted African cat famous for being the fastest land animal? Answer in one word.',
            'height': 'What is the height of the large spotted African cat famous for being the fastest land animal? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which large spotted African wild cat is famous for being the fastest land animal in the world?',
            'What spotted wild cat has distinctive black tear-like streaks on its face and is renowned as the fastest land animal?'
        ]
    },
    {
        'animal': 'Peacock',
        'accepted_answers': ['Peacock', 'Peafowl', 'Pavo'],
        'description': 'A brightly colored male bird that displays a massive, iridescent fan of tail feathers, using a train of eye-spotted feathers for elaborate courtship displays.',
        'unique_features': [
            'massive iridescent fan of tail feathers',
            'eye-spotted train',
        ],
        'shared_features': [
            ['bird', 'brightly colored male', 'massive iridescent fan of tail feathers'],
            ['bird', 'brightly colored male', 'massive iridescent fan of tail feathers', 'eye-spotted train'],
            ['bird', 'brightly colored male', 'massive iridescent fan of tail feathers', 'elaborate courtship display'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4 (2 legs, 2 wings)', 'accepted': ['4', 'four'], 'note': '2 legs and 2 wings'},
            'color': {'type': 'set', 'display': 'Iridescent blue and green', 'accepted': ['iridescent', 'blue', 'green', 'blue-green', 'teal']},
            'weight': {'type': 'range', 'display': '4 - 6 kg', 'min': 4, 'max': 6, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '10 - 25 years', 'min': 10, 'max': 25, 'unit': 'years'},
            'height': {'type': 'range', 'display': '1.0 - 1.2 m', 'min': 1.0, 'max': 1.2, 'unit': 'm', 'measure': 'standing height, excluding train'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs in total does the brightly colored male bird that displays an iridescent fan of tail feathers have? Answer in one word.',
            'color': 'What is the color of the brightly colored male bird that displays an iridescent fan of tail feathers? Answer in one word.',
            'weight': 'What is the weight of the brightly colored male bird that displays an iridescent fan of tail feathers? Answer in one word.',
            'lifespan': 'What is the lifespan of the brightly colored male bird that displays an iridescent fan of tail feathers? Answer in one word.',
            'height': 'What is the height of the brightly colored male bird that displays an iridescent fan of tail feathers? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which specific male bird is widely famous for displaying a massive, iridescent fan of eye-spotted tail feathers?',
            'What specific brightly colored male bird shakes an enormous fan of tail feathers during elaborate courtship displays?'
        ]
    },
    {
        'animal': 'Porcupine',
        'accepted_answers': ['Porcupine'],
        'description': 'A large, slow-moving rodent covered in sharp, detachable quills, which lodges its barbed defensive spines into the skin of attacking predators.',
        'unique_features': [
            'covered in sharp detachable quills',
        ],
        'shared_features': [
            ['rodent', 'large', 'slow-moving', 'covered in sharp detachable quills'],
            ['rodent', 'covered in sharp detachable quills', 'barbed spines lodge in predators'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Dark brown or black with white-tipped quills', 'accepted': ['dark brown', 'brown', 'black', 'white']},
            'weight': {'type': 'range', 'display': '5 - 16 kg', 'min': 5, 'max': 16, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '10 - 15 years', 'min': 10, 'max': 15, 'unit': 'years'},
            'height': {'type': 'range', 'display': '20 - 35 cm', 'min': 20, 'max': 35, 'unit': 'cm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the large, slow-moving rodent covered in sharp, detachable quills have? Answer in one word.',
            'color': 'What is the color of the large, slow-moving rodent covered in sharp, detachable quills? Answer in one word.',
            'weight': 'What is the weight of the large, slow-moving rodent covered in sharp, detachable quills? Answer in one word.',
            'lifespan': 'What is the lifespan of the large, slow-moving rodent covered in sharp, detachable quills? Answer in one word.',
            'height': 'What is the height of the large, slow-moving rodent covered in sharp, detachable quills? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which large, slow-moving rodent defends itself with an array of sharp, detachable barbed quills?',
            'What large quill-bearing rodent can easily lodge its sharp, defensive spines into the skin of attacking predators?'
        ]
    },
    {
        'animal': 'Armadillo',
        'accepted_answers': ['Armadillo', 'Dasypodidae'],
        'description': 'A New World mammal with a protective leathery armor shell made of overlapping bony plates, which rolls into a tight protective ball when threatened by predators.',
        'unique_features': [
            'leathery armor shell',
            'overlapping bony plates',
        ],
        'shared_features': [
            ['mammal', 'native to the Americas', 'leathery armor shell'],
            ['mammal', 'native to the Americas', 'leathery armor shell', 'overlapping bony plates'],
            ['mammal', 'native to the Americas', 'leathery armor shell', 'rolls into a protective ball when threatened'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Greyish-brown to pinkish', 'accepted': ['greyish-brown', 'grey', 'gray', 'brown', 'pinkish', 'pink']},
            'weight': {'type': 'range', 'display': '2.5 - 60 kg', 'min': 2.5, 'max': 60, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '12 - 15 years', 'min': 12, 'max': 15, 'unit': 'years'},
            'height': {'type': 'range', 'display': '15 - 60 cm', 'min': 15, 'max': 60, 'unit': 'cm', 'measure': 'body length'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the New World mammal with a protective leathery armor shell have? Answer in one word.',
            'color': 'What is the color of the New World mammal with a protective leathery armor shell? Answer in one word.',
            'weight': 'What is the weight of the New World mammal with a protective leathery armor shell? Answer in one word.',
            'lifespan': 'What is the lifespan of the New World mammal with a protective leathery armor shell? Answer in one word.',
            'height': 'What is the length of the New World mammal with a protective leathery armor shell? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which New World mammal has a protective leathery armor shell made of overlapping bony plates?',
            'What armored mammal from the Americas is known to roll into a tight protective ball when threatened by predators?'
        ]
    },
    {
        'animal': 'Wombat',
        'accepted_answers': ['Wombat', 'Common wombat', 'Vombatidae'],
        'description': 'A stout, burrowing Australian marsupial uniquely famous for naturally producing cube-shaped feces, with a backward-facing pouch to prevent dirt from entering while digging.',
        'unique_features': [
            'produces cube-shaped feces',
            'backward-facing pouch',
        ],
        'shared_features': [
            ['marsupial', 'native to Australia', 'burrowing', 'produces cube-shaped feces'],
            ['marsupial', 'native to Australia', 'stout and short-legged', 'burrowing', 'backward-facing pouch'],
        ],
        'attributes': {
            'number_of_limbs': {'type': 'set', 'display': '4', 'accepted': ['4', 'four']},
            'color': {'type': 'set', 'display': 'Brown, grey, or black', 'accepted': ['brown', 'grey', 'gray', 'black']},
            'weight': {'type': 'range', 'display': '20 - 35 kg', 'min': 20, 'max': 35, 'unit': 'kg'},
            'lifespan': {'type': 'range', 'display': '15 - 20 years', 'min': 15, 'max': 20, 'unit': 'years'},
            'height': {'type': 'range', 'display': '25 - 35 cm', 'min': 25, 'max': 35, 'unit': 'cm', 'measure': 'shoulder height'},
        },
        'attribute_questions': {
            'number_of_limbs': 'How many limbs does the burrowing Australian marsupial uniquely famous for naturally producing cube-shaped feces have? Answer in one word.',
            'color': 'What is the color of the burrowing Australian marsupial uniquely famous for naturally producing cube-shaped feces? Answer in one word.',
            'weight': 'What is the weight of the burrowing Australian marsupial uniquely famous for naturally producing cube-shaped feces? Answer in one word.',
            'lifespan': 'What is the lifespan of the burrowing Australian marsupial uniquely famous for naturally producing cube-shaped feces? Answer in one word.',
            'height': 'What is the height of the burrowing Australian marsupial uniquely famous for naturally producing cube-shaped feces? Answer in one word.'
        },
        'jeopardy_questions': [
            'Which stout, burrowing Australian marsupial (belonging to the family Vombatidae) is uniquely famous for naturally producing dry, cube-shaped feces?',
            'What stout, short-legged Australian marsupial has a backward-facing pouch to prevent dirt from entering while digging?'
        ]
    }
]

ANIMALS_DICT = {animal['animal'].lower(): animal for animal in ANIMALS}


def main():
    json.dump(ANIMALS, open('animals.json', 'w'), indent=2, ensure_ascii=False)
    #sys.stdout.write('\n')


if __name__ == '__main__':
    main()
