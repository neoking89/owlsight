from owlsight.utils.helper_functions import extract_square_bracket_tags


def test_extract_square_bracket_tags():
    # Example 1
    input_1 = '[[load:path-to-model1.json]] How much stomachs has a cow?'
    expected_1 = [{'path': 'path-to-model1.json'}, 'How much stomachs has a cow?']
    assert extract_square_bracket_tags(input_1, tag="load", key="path") == expected_1
    
    # Example 2
    input_2 = '[[load:path-to-model1.json]][[image:path-to-image.jpg]]'
    expected_2 = [{'path': 'path-to-model1.json'}, '[[image:path-to-image.jpg]]']
    assert extract_square_bracket_tags(input_2, tag="load", key="path") == expected_2
    
    # Example 3
    input_3 = '[[load:path-to-model1.json]] [[image:path-to-image.jpg]] [[load:path-to-model2.json]] Some question about the output'
    expected_3 = [
        {'path': 'path-to-model1.json'},
        '[[image:path-to-image.jpg]]',
        {'path': 'path-to-model2.json'},
        'Some question about the output'
    ]
    assert extract_square_bracket_tags(input_3, tag="load", key="path") == expected_3
    
    # Example 4
    input_4 = '[[load:path-to-model1.json]] [[load:path-to-model2.json]]'
    expected_4 = [{'path': 'path-to-model1.json'}, {'path': 'path-to-model2.json'}]
    assert extract_square_bracket_tags(input_4, tag="load", key="path") == expected_4
    
    # Example 5
    input_5 = '[[load:path-to-model1.json]] [[image:path-to-image.jpg]] [[load:path-to-model2.json]] [[load:path-to-model3.json]]'
    expected_5 = [
        {'path': 'path-to-model1.json'},
        '[[image:path-to-image.jpg]]',
        {'path': 'path-to-model2.json'},
        {'path': 'path-to-model3.json'}
    ]
    assert extract_square_bracket_tags(input_5, tag="load", key="path") == expected_5
    
    # Example 6
    input_6 = '[[load:path-to-model1.json]] [[load:path-to-model2.json]] [[load:path-to-model3.json]]'
    expected_6 = [
        {'path': 'path-to-model1.json'},
        {'path': 'path-to-model2.json'},
        {'path': 'path-to-model3.json'}
    ]
    assert extract_square_bracket_tags(input_6, tag="load", key="path") == expected_6
    
    # Example 7
    input_7 = '[[load:path-to-model1.json]] [[image:path-to-image.jpg]] [[load:path-to-model2.json]] [[image:path-to-image2.jpg]]'
    expected_7 = [
        {'path': 'path-to-model1.json'},
        '[[image:path-to-image.jpg]]',
        {'path': 'path-to-model2.json'},
        '[[image:path-to-image2.jpg]]'
    ]
    assert extract_square_bracket_tags(input_7, tag="load", key="path") == expected_7
    
    # Example 8
    input_8 = '[[load:path-to-model1.json]] [[load:path-to-model2.json]] [[image:path-to-image.jpg]] [[load:path-to-model3.json]]'
    expected_8 = [
        {'path': 'path-to-model1.json'},
        {'path': 'path-to-model2.json'},
        '[[image:path-to-image.jpg]]',
        {'path': 'path-to-model3.json'}
    ]
    assert extract_square_bracket_tags(input_8, tag="load", key="path") == expected_8
    
    # Example 9
    input_9 = '[[load:path-to-model1.json]] [[image:path-to-image.jpg]] [[image:path-to-image2.jpg]] [[load:path-to-model2.json]]'
    expected_9 = [
        {'path': 'path-to-model1.json'},
        '[[image:path-to-image.jpg]] [[image:path-to-image2.jpg]]',
        {'path': 'path-to-model2.json'}
    ]
    assert extract_square_bracket_tags(input_9, tag="load", key="path") == expected_9
    
    # Example 10
    input_10 = '[[load:path-to-model1.json]] [[load:path-to-model2.json]] [[load:path-to-model3.json]] [[load:path-to-model4.json]]'
    expected_10 = [
        {'path': 'path-to-model1.json'},
        {'path': 'path-to-model2.json'},
        {'path': 'path-to-model3.json'},
        {'path': 'path-to-model4.json'}
    ]
    assert extract_square_bracket_tags(input_10, tag="load", key="path") == expected_10
