
====================================================================================================
0  model allenai/Olmo-3-7B-Instruct on cuda
====================================================================================================
  layers 32 -> states h00..h31, deltas d01..d31; hidden 4096; dtype torch.bfloat16
  FWD_BATCH 32, GEN_BATCH 64, MAX_NEW_TOKENS 48
  built 19630 rows from animals.json

====================================================================================================
1  prompt token tables (check every marker by eye)
====================================================================================================

--- random 3522 | Sloth|attribute|number_of_limbs|1|2
  question: What number of limbs does the mammal of the Neotropical canopy, famed for its lethargy, whose shaggy coat grows algae and shelters moths have? Answer in one word.
  templated: '<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nWhat number of limbs does the mammal of the Neotropical canopy, famed for its lethargy, whose shaggy coat grows algae and shelters moths have? Answer in one word.<|im_end|>\n<|im_start|>assistant\n'
   idx  coord          id  token                     markers
     0  sys_00     100264  '<|im_start|>'            seq_begin
     1  sys_01       9125  'system'                  
     2  sys_02        198  '\n'                      
     3  sys_03       2675  'You'                     
     4  sys_04        527  ' are'                    
     5  sys_05        264  ' a'                      
     6  sys_06      11190  ' helpful'                
     7  sys_07        734  ' function'               
     8  sys_08       1824  '-c'                      
     9  sys_09      17157  'alling'                  
    10  sys_10      15592  ' AI'                     
    11  sys_11      18328  ' assistant'              
    12  sys_12         13  '.'                       
    13  sys_13       1472  ' You'                    
    14  sys_14        656  ' do'                     
    15  sys_15        539  ' not'                    
    16  sys_16       5131  ' currently'              
    17  sys_17        617  ' have'                   
    18  sys_18       2680  ' access'                 
    19  sys_19        311  ' to'                     
    20  sys_20        904  ' any'                    
    21  sys_21       5865  ' functions'              
    22  sys_22         13  '.'                       
    23  sys_23        220  ' '                       
    24  sys_24     100266  '<functions>'             
    25  sys_25     100267  '</functions>'            
    26  sys_26     100265  '<|im_end|>'              
    27  sys_27        198  '\n'                      
    28  sys_28     100264  '<|im_start|>'            
    29  sys_29        882  'user'                    
    30  sys_30        198  '\n'                      
    31  pre_00       3923  'What'                    
    32  pre_01       1396  ' number'                 
    33  pre_02        315  ' of'                     
    34  pre_03      49695  ' limbs'                  
    35  pre_04       1587  ' does'                   
    36  id_00         279  ' the'                    
    37  id_01       36041  ' mamm'                   
    38  id_02         278  'al'                      
    39  id_03         315  ' of'                     
    40  id_04         279  ' the'                    
    41  id_05        4275  ' Ne'                     
    42  id_06         354  'ot'                      
    43  id_07       51172  'ropical'                 
    44  id_08       64071  ' canopy'                 
    45  id_09          11  ','                       
    46  id_10       61403  ' famed'                  
    47  id_11         369  ' for'                    
    48  id_12        1202  ' its'                    
    49  id_13       98522  ' leth'                   
    50  id_14         867  'arg'                     
    51  id_15          88  'y'                       
    52  id_16          11  ','                       
    53  id_17        6832  ' whose'                  
    54  id_18         559  ' sh'                     
    55  id_19         351  'ag'                      
    56  id_20        4680  'gy'                      
    57  id_21       23724  ' coat'                   
    58  id_22       28815  ' grows'                  
    59  id_23       68951  ' algae'                  
    60  id_24         323  ' and'                    
    61  id_25       52888  ' shelters'               
    62  id_26        4647  ' mo'                     
    63  id_27       17323  'ths'                     ident_last
    64  post_00       617  ' have'                   
    65  post_01        30  '?'                       
    66  post_02     22559  ' Answer'                 
    67  post_03       304  ' in'                     
    68  post_04       832  ' one'                    
    69  post_05      3492  ' word'                   
    70  post_06        13  '.'                       
    71  tmpl_00    100265  '<|im_end|>'              
    72  tmpl_01       198  '\n'                      end_user
    73  tmpl_02    100264  '<|im_start|>'            assistant_tmp_begin
    74  tmpl_03     78191  'assistant'               
    75  tmpl_04       198  '\n'                      assistant_tmp_end
  segment lengths: {'sys': 31, 'pre': 5, 'id': 28, 'post': 7, 'tmpl': 5}
  [OK] coordinates unique, one per token
  [OK] ident_last is the last id token
  [OK] assistant_tmp_begin directly follows end_user
  [OK] assistant_tmp_end is the last prompt token (= v1 prompt_last)

--- random 15926 | Cheetah|attribute|lifespan|3|6
  question: How long-lived is the fastest land animal, an African cat whose spine flexes so far that it is airborne for much of each stride on average? Answer with a number of years.
  templated: '<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nHow long-lived is the fastest land animal, an African cat whose spine flexes so far that it is airborne for much of each stride on average? Answer with a number of years.<|im_end|>\n<|im_start|>assistant\n'
   idx  coord          id  token                     markers
     0  sys_00     100264  '<|im_start|>'            seq_begin
     1  sys_01       9125  'system'                  
     2  sys_02        198  '\n'                      
     3  sys_03       2675  'You'                     
     4  sys_04        527  ' are'                    
     5  sys_05        264  ' a'                      
     6  sys_06      11190  ' helpful'                
     7  sys_07        734  ' function'               
     8  sys_08       1824  '-c'                      
     9  sys_09      17157  'alling'                  
    10  sys_10      15592  ' AI'                     
    11  sys_11      18328  ' assistant'              
    12  sys_12         13  '.'                       
    13  sys_13       1472  ' You'                    
    14  sys_14        656  ' do'                     
    15  sys_15        539  ' not'                    
    16  sys_16       5131  ' currently'              
    17  sys_17        617  ' have'                   
    18  sys_18       2680  ' access'                 
    19  sys_19        311  ' to'                     
    20  sys_20        904  ' any'                    
    21  sys_21       5865  ' functions'              
    22  sys_22         13  '.'                       
    23  sys_23        220  ' '                       
    24  sys_24     100266  '<functions>'             
    25  sys_25     100267  '</functions>'            
    26  sys_26     100265  '<|im_end|>'              
    27  sys_27        198  '\n'                      
    28  sys_28     100264  '<|im_start|>'            
    29  sys_29        882  'user'                    
    30  sys_30        198  '\n'                      
    31  pre_00       4438  'How'                     
    32  pre_01       1317  ' long'                   
    33  pre_02      62954  '-lived'                  
    34  pre_03        374  ' is'                     
    35  id_00         279  ' the'                    
    36  id_01       26731  ' fastest'                
    37  id_02        4363  ' land'                   
    38  id_03       10065  ' animal'                 
    39  id_04          11  ','                       
    40  id_05         459  ' an'                     
    41  id_06       11904  ' African'                
    42  id_07        8415  ' cat'                    
    43  id_08        6832  ' whose'                  
    44  id_09       35776  ' spine'                  
    45  id_10        5882  ' flex'                   
    46  id_11         288  'es'                      
    47  id_12         779  ' so'                     
    48  id_13        3117  ' far'                    
    49  id_14         430  ' that'                   
    50  id_15         433  ' it'                     
    51  id_16         374  ' is'                     
    52  id_17       70863  ' airborne'               
    53  id_18         369  ' for'                    
    54  id_19        1790  ' much'                   
    55  id_20         315  ' of'                     
    56  id_21        1855  ' each'                   
    57  id_22       19782  ' stride'                 ident_last
    58  post_00       389  ' on'                     
    59  post_01      5578  ' average'                
    60  post_02        30  '?'                       
    61  post_03     22559  ' Answer'                 
    62  post_04       449  ' with'                   
    63  post_05       264  ' a'                      
    64  post_06      1396  ' number'                 
    65  post_07       315  ' of'                     
    66  post_08      1667  ' years'                  
    67  post_09        13  '.'                       
    68  tmpl_00    100265  '<|im_end|>'              
    69  tmpl_01       198  '\n'                      end_user
    70  tmpl_02    100264  '<|im_start|>'            assistant_tmp_begin
    71  tmpl_03     78191  'assistant'               
    72  tmpl_04       198  '\n'                      assistant_tmp_end
  segment lengths: {'sys': 31, 'pre': 4, 'id': 23, 'post': 10, 'tmpl': 5}
  [OK] coordinates unique, one per token
  [OK] ident_last is the last id token
  [OK] assistant_tmp_begin directly follows end_user
  [OK] assistant_tmp_end is the last prompt token (= v1 prompt_last)

--- random 1681 | Kangaroo|attribute|offspring_count|7|1
  question: How many offspring does the big-footed Australian marsupial that boxes with its forearms when challenged have in a single reproductive event? Answer with a number or a range.
  templated: '<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nHow many offspring does the big-footed Australian marsupial that boxes with its forearms when challenged have in a single reproductive event? Answer with a number or a range.<|im_end|>\n<|im_start|>assistant\n'
   idx  coord          id  token                     markers
     0  sys_00     100264  '<|im_start|>'            seq_begin
     1  sys_01       9125  'system'                  
     2  sys_02        198  '\n'                      
     3  sys_03       2675  'You'                     
     4  sys_04        527  ' are'                    
     5  sys_05        264  ' a'                      
     6  sys_06      11190  ' helpful'                
     7  sys_07        734  ' function'               
     8  sys_08       1824  '-c'                      
     9  sys_09      17157  'alling'                  
    10  sys_10      15592  ' AI'                     
    11  sys_11      18328  ' assistant'              
    12  sys_12         13  '.'                       
    13  sys_13       1472  ' You'                    
    14  sys_14        656  ' do'                     
    15  sys_15        539  ' not'                    
    16  sys_16       5131  ' currently'              
    17  sys_17        617  ' have'                   
    18  sys_18       2680  ' access'                 
    19  sys_19        311  ' to'                     
    20  sys_20        904  ' any'                    
    21  sys_21       5865  ' functions'              
    22  sys_22         13  '.'                       
    23  sys_23        220  ' '                       
    24  sys_24     100266  '<functions>'             
    25  sys_25     100267  '</functions>'            
    26  sys_26     100265  '<|im_end|>'              
    27  sys_27        198  '\n'                      
    28  sys_28     100264  '<|im_start|>'            
    29  sys_29        882  'user'                    
    30  sys_30        198  '\n'                      
    31  pre_00       4438  'How'                     
    32  pre_01       1690  ' many'                   
    33  pre_02      46471  ' offspring'              
    34  pre_03       1587  ' does'                   
    35  id_00         279  ' the'                    
    36  id_01        2466  ' big'                    
    37  id_02       21117  '-foot'                   
    38  id_03         291  'ed'                      
    39  id_04       13673  ' Australian'             
    40  id_05       48449  ' mars'                   
    41  id_06         455  'up'                      
    42  id_07         532  'ial'                     
    43  id_08         430  ' that'                   
    44  id_09       15039  ' boxes'                  
    45  id_10         449  ' with'                   
    46  id_11        1202  ' its'                    
    47  id_12        2291  ' fore'                   
    48  id_13       16293  'arms'                    
    49  id_14         994  ' when'                   
    50  id_15       29991  ' challenged'             ident_last
    51  post_00       617  ' have'                   
    52  post_01       304  ' in'                     
    53  post_02       264  ' a'                      
    54  post_03      3254  ' single'                 
    55  post_04     42889  ' reproductive'           
    56  post_05      1567  ' event'                  
    57  post_06        30  '?'                       
    58  post_07     22559  ' Answer'                 
    59  post_08       449  ' with'                   
    60  post_09       264  ' a'                      
    61  post_10      1396  ' number'                 
    62  post_11       477  ' or'                     
    63  post_12       264  ' a'                      
    64  post_13      2134  ' range'                  
    65  post_14        13  '.'                       
    66  tmpl_00    100265  '<|im_end|>'              
    67  tmpl_01       198  '\n'                      end_user
    68  tmpl_02    100264  '<|im_start|>'            assistant_tmp_begin
    69  tmpl_03     78191  'assistant'               
    70  tmpl_04       198  '\n'                      assistant_tmp_end
  segment lengths: {'sys': 31, 'pre': 4, 'id': 16, 'post': 15, 'tmpl': 5}
  [OK] coordinates unique, one per token
  [OK] ident_last is the last id token
  [OK] assistant_tmp_begin directly follows end_user
  [OK] assistant_tmp_end is the last prompt token (= v1 prompt_last)

--- random 4648 | Elephant|attribute|lifespan|5|8
  question: For how many years does the tusked mammal with a long trunk that mourns its dead and revisits their bones normally live? Answer with a number of years.
  templated: '<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nFor how many years does the tusked mammal with a long trunk that mourns its dead and revisits their bones normally live? Answer with a number of years.<|im_end|>\n<|im_start|>assistant\n'
   idx  coord          id  token                     markers
     0  sys_00     100264  '<|im_start|>'            seq_begin
     1  sys_01       9125  'system'                  
     2  sys_02        198  '\n'                      
     3  sys_03       2675  'You'                     
     4  sys_04        527  ' are'                    
     5  sys_05        264  ' a'                      
     6  sys_06      11190  ' helpful'                
     7  sys_07        734  ' function'               
     8  sys_08       1824  '-c'                      
     9  sys_09      17157  'alling'                  
    10  sys_10      15592  ' AI'                     
    11  sys_11      18328  ' assistant'              
    12  sys_12         13  '.'                       
    13  sys_13       1472  ' You'                    
    14  sys_14        656  ' do'                     
    15  sys_15        539  ' not'                    
    16  sys_16       5131  ' currently'              
    17  sys_17        617  ' have'                   
    18  sys_18       2680  ' access'                 
    19  sys_19        311  ' to'                     
    20  sys_20        904  ' any'                    
    21  sys_21       5865  ' functions'              
    22  sys_22         13  '.'                       
    23  sys_23        220  ' '                       
    24  sys_24     100266  '<functions>'             
    25  sys_25     100267  '</functions>'            
    26  sys_26     100265  '<|im_end|>'              
    27  sys_27        198  '\n'                      
    28  sys_28     100264  '<|im_start|>'            
    29  sys_29        882  'user'                    
    30  sys_30        198  '\n'                      
    31  pre_00       2520  'For'                     
    32  pre_01       1268  ' how'                    
    33  pre_02       1690  ' many'                   
    34  pre_03       1667  ' years'                  
    35  pre_04       1587  ' does'                   
    36  id_00         279  ' the'                    
    37  id_01         259  ' t'                      
    38  id_02       33879  'usk'                     
    39  id_03         291  'ed'                      
    40  id_04       36041  ' mamm'                   
    41  id_05         278  'al'                      
    42  id_06         449  ' with'                   
    43  id_07         264  ' a'                      
    44  id_08        1317  ' long'                   
    45  id_09       38411  ' trunk'                  
    46  id_10         430  ' that'                   
    47  id_11       60234  ' mourn'                  
    48  id_12          82  's'                       
    49  id_13        1202  ' its'                    
    50  id_14        5710  ' dead'                   
    51  id_15         323  ' and'                    
    52  id_16       17951  ' revis'                  
    53  id_17        1220  'its'                     
    54  id_18         872  ' their'                  
    55  id_19       25896  ' bones'                  ident_last
    56  post_00     14614  ' normally'               
    57  post_01      3974  ' live'                   
    58  post_02        30  '?'                       
    59  post_03     22559  ' Answer'                 
    60  post_04       449  ' with'                   
    61  post_05       264  ' a'                      
    62  post_06      1396  ' number'                 
    63  post_07       315  ' of'                     
    64  post_08      1667  ' years'                  
    65  post_09        13  '.'                       
    66  tmpl_00    100265  '<|im_end|>'              
    67  tmpl_01       198  '\n'                      end_user
    68  tmpl_02    100264  '<|im_start|>'            assistant_tmp_begin
    69  tmpl_03     78191  'assistant'               
    70  tmpl_04       198  '\n'                      assistant_tmp_end
  segment lengths: {'sys': 31, 'pre': 5, 'id': 20, 'post': 10, 'tmpl': 5}
  [OK] coordinates unique, one per token
  [OK] ident_last is the last id token
  [OK] assistant_tmp_begin directly follows end_user
  [OK] assistant_tmp_end is the last prompt token (= v1 prompt_last)

--- random 3560 | Sloth|attribute|number_of_limbs|6|0
  question: How many limbs does the long-clawed mammal of the Neotropical canopy, famed for its lethargy, that nonetheless swims surprisingly well have? Answer in one word.
  templated: '<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nHow many limbs does the long-clawed mammal of the Neotropical canopy, famed for its lethargy, that nonetheless swims surprisingly well have? Answer in one word.<|im_end|>\n<|im_start|>assistant\n'
   idx  coord          id  token                     markers
     0  sys_00     100264  '<|im_start|>'            seq_begin
     1  sys_01       9125  'system'                  
     2  sys_02        198  '\n'                      
     3  sys_03       2675  'You'                     
     4  sys_04        527  ' are'                    
     5  sys_05        264  ' a'                      
     6  sys_06      11190  ' helpful'                
     7  sys_07        734  ' function'               
     8  sys_08       1824  '-c'                      
     9  sys_09      17157  'alling'                  
    10  sys_10      15592  ' AI'                     
    11  sys_11      18328  ' assistant'              
    12  sys_12         13  '.'                       
    13  sys_13       1472  ' You'                    
    14  sys_14        656  ' do'                     
    15  sys_15        539  ' not'                    
    16  sys_16       5131  ' currently'              
    17  sys_17        617  ' have'                   
    18  sys_18       2680  ' access'                 
    19  sys_19        311  ' to'                     
    20  sys_20        904  ' any'                    
    21  sys_21       5865  ' functions'              
    22  sys_22         13  '.'                       
    23  sys_23        220  ' '                       
    24  sys_24     100266  '<functions>'             
    25  sys_25     100267  '</functions>'            
    26  sys_26     100265  '<|im_end|>'              
    27  sys_27        198  '\n'                      
    28  sys_28     100264  '<|im_start|>'            
    29  sys_29        882  'user'                    
    30  sys_30        198  '\n'                      
    31  pre_00       4438  'How'                     
    32  pre_01       1690  ' many'                   
    33  pre_02      49695  ' limbs'                  
    34  pre_03       1587  ' does'                   
    35  id_00         279  ' the'                    
    36  id_01        1317  ' long'                   
    37  id_02       31717  '-cl'                     
    38  id_03         675  'aw'                      
    39  id_04         291  'ed'                      
    40  id_05       36041  ' mamm'                   
    41  id_06         278  'al'                      
    42  id_07         315  ' of'                     
    43  id_08         279  ' the'                    
    44  id_09        4275  ' Ne'                     
    45  id_10         354  'ot'                      
    46  id_11       51172  'ropical'                 
    47  id_12       64071  ' canopy'                 
    48  id_13          11  ','                       
    49  id_14       61403  ' famed'                  
    50  id_15         369  ' for'                    
    51  id_16        1202  ' its'                    
    52  id_17       98522  ' leth'                   
    53  id_18         867  'arg'                     
    54  id_19          88  'y'                       
    55  id_20          11  ','                       
    56  id_21         430  ' that'                   
    57  id_22       38913  ' nonetheless'            
    58  id_23       81736  ' swims'                  
    59  id_24       29392  ' surprisingly'           
    60  id_25        1664  ' well'                   ident_last
    61  post_00       617  ' have'                   
    62  post_01        30  '?'                       
    63  post_02     22559  ' Answer'                 
    64  post_03       304  ' in'                     
    65  post_04       832  ' one'                    
    66  post_05      3492  ' word'                   
    67  post_06        13  '.'                       
    68  tmpl_00    100265  '<|im_end|>'              
    69  tmpl_01       198  '\n'                      end_user
    70  tmpl_02    100264  '<|im_start|>'            assistant_tmp_begin
    71  tmpl_03     78191  'assistant'               
    72  tmpl_04       198  '\n'                      assistant_tmp_end
  segment lengths: {'sys': 31, 'pre': 4, 'id': 26, 'post': 7, 'tmpl': 5}
  [OK] coordinates unique, one per token
  [OK] ident_last is the last id token
  [OK] assistant_tmp_begin directly follows end_user
  [OK] assistant_tmp_end is the last prompt token (= v1 prompt_last)
  [OK] sys tokens identical across the 5 rows
  [OK] tmpl tokens identical across the 5 rows

====================================================================================================
2  generation: batched (prepare) vs one at a time
====================================================================================================
generate 5/5

--- random 3522
  answer 'Four'  correct=True  capped=False  gen_ids [28070, 100257]
  [OK] unbatched generation gives the same ids
  answer part of the token table:
    74  tmpl_03   'assistant'               
    75  tmpl_04   '\n'                      assistant_tmp_end
    76  ans_00    'Four'                    answer_first answer_last
    77  ans_01    '<|endoftext|>'           last
  [OK] last is the final token
  [OK] answer_last precedes the end token

--- random 15926
  answer 'The fastest land animal is the cheetah. While the question refers to a "African cat" with a highly flexible spine, this description most closely matches the cheetah (Acinonyx jubatus), which can reach speeds'  correct=False  capped=True  gen_ids [791, 26731, 4363, 10065, 374, 279, 3091, 295, 1494, 13, 6104, 279, 3488, 19813, 311, 264, 330, 32, 48893, 8415, 1, 449, 264, 7701, 19303, 35776, 11, 420, 4096, 1455, 15499, 9248, 279, 3091, 295, 1494, 320, 11916, 258, 3633, 87, 97443, 1015, 705, 902, 649, 5662, 25753]
  [OK] unbatched generation gives the same ids
  answer part of the token table:
    71  tmpl_03   'assistant'               
    72  tmpl_04   '\n'                      assistant_tmp_end
    73  ans_00    'The'                     answer_first
    74  ans_01    ' fastest'                
    75  ans_02    ' land'                   
    76  ans_03    ' animal'                 
    77  ans_04    ' is'                     
    78  ans_05    ' the'                    
    79  ans_06    ' che'                    
    80  ans_07    'et'                      
    81  ans_08    'ah'                      
    82  ans_09    '.'                       
    83  ans_10    ' While'                  
    84  ans_11    ' the'                    
    85  ans_12    ' question'               
    86  ans_13    ' refers'                 
    87  ans_14    ' to'                     
    88  ans_15    ' a'                      
    89  ans_16    ' "'                      
    90  ans_17    'A'                       
    91  ans_18    'frican'                  
    92  ans_19    ' cat'                    
    93  ans_20    '"'                       
    94  ans_21    ' with'                   
    95  ans_22    ' a'                      
    96  ans_23    ' highly'                 
    97  ans_24    ' flexible'               
    98  ans_25    ' spine'                  
    99  ans_26    ','                       
   100  ans_27    ' this'                   
   101  ans_28    ' description'            
   102  ans_29    ' most'                   
   103  ans_30    ' closely'                
   104  ans_31    ' matches'                
   105  ans_32    ' the'                    
   106  ans_33    ' che'                    
   107  ans_34    'et'                      
   108  ans_35    'ah'                      
   109  ans_36    ' ('                      
   110  ans_37    'Ac'                      
   111  ans_38    'in'                      
   112  ans_39    'ony'                     
   113  ans_40    'x'                       
   114  ans_41    ' jub'                    
   115  ans_42    'atus'                    
   116  ans_43    '),'                      
   117  ans_44    ' which'                  
   118  ans_45    ' can'                    
   119  ans_46    ' reach'                  
   120  ans_47    ' speeds'                 answer_last last
  [OK] last is the final token
  [OK] answer_last == last
  [WARN] no end token within 48 tokens: the answer may be cut off, answer_last/last sit on the cap, and 'correct' may be wrong

--- random 1681
  answer "The big-footed Australian marsupial you're referring to is likely the **Tasmanian devil** (*Sarcophilus harrisii*), though it is not known for boxing with its forearms. However, among mars"  correct=False  capped=True  gen_ids [791, 2466, 21117, 291, 13673, 48449, 455, 532, 499, 2351, 22797, 311, 374, 4461, 279, 3146, 51, 300, 1543, 1122, 38736, 334, 4712, 50, 8362, 98635, 355, 305, 59422, 3893, 9, 705, 3582, 433, 374, 539, 3967, 369, 46329, 449, 1202, 2291, 16293, 13, 4452, 11, 4315, 48449]
  [WARN] unbatched generation diverges at answer token 16 ('T' vs 'per'); identical before: "The big-footed Australian marsupial you're referring to is likely the **"
  answer part of the token table:
    69  tmpl_03   'assistant'               
    70  tmpl_04   '\n'                      assistant_tmp_end
    71  ans_00    'The'                     answer_first
    72  ans_01    ' big'                    
    73  ans_02    '-foot'                   
    74  ans_03    'ed'                      
    75  ans_04    ' Australian'             
    76  ans_05    ' mars'                   
    77  ans_06    'up'                      
    78  ans_07    'ial'                     
    79  ans_08    ' you'                    
    80  ans_09    "'re"                     
    81  ans_10    ' referring'              
    82  ans_11    ' to'                     
    83  ans_12    ' is'                     
    84  ans_13    ' likely'                 
    85  ans_14    ' the'                    
    86  ans_15    ' **'                     
    87  ans_16    'T'                       
    88  ans_17    'as'                      
    89  ans_18    'man'                     
    90  ans_19    'ian'                     
    91  ans_20    ' devil'                  
    92  ans_21    '**'                      
    93  ans_22    ' (*'                     
    94  ans_23    'S'                       
    95  ans_24    'arc'                     
    96  ans_25    'ophil'                   
    97  ans_26    'us'                      
    98  ans_27    ' h'                      
    99  ans_28    'arris'                   
   100  ans_29    'ii'                      
   101  ans_30    '*'                       
   102  ans_31    '),'                      
   103  ans_32    ' though'                 
   104  ans_33    ' it'                     
   105  ans_34    ' is'                     
   106  ans_35    ' not'                    
   107  ans_36    ' known'                  
   108  ans_37    ' for'                    
   109  ans_38    ' boxing'                 
   110  ans_39    ' with'                   
   111  ans_40    ' its'                    
   112  ans_41    ' fore'                   
   113  ans_42    'arms'                    
   114  ans_43    '.'                       
   115  ans_44    ' However'                
   116  ans_45    ','                       
   117  ans_46    ' among'                  
   118  ans_47    ' mars'                   answer_last last
  [OK] last is the final token
  [OK] answer_last == last
  [WARN] no end token within 48 tokens: the answer may be cut off, answer_last/last sit on the cap, and 'correct' may be wrong

--- random 4648
  answer 'Elephants typically live for 60 to 70 years in the wild, though their maximum lifespan can occasionally reach up to 80 years. Given the question refers to a "tusked mammal with a long trunk that mourns its'  correct=True  capped=True  gen_ids [46439, 53667, 11383, 3974, 369, 220, 1399, 311, 220, 2031, 1667, 304, 279, 8545, 11, 3582, 872, 7340, 61961, 649, 23781, 5662, 709, 311, 220, 1490, 1667, 13, 16644, 279, 3488, 19813, 311, 264, 330, 83, 33879, 291, 36041, 278, 449, 264, 1317, 38411, 430, 60234, 82, 1202]
  [WARN] unbatched generation diverges at answer token 27 ('.' vs ' in'); identical before: 'Elephants typically live for 60 to 70 years in the wild, though their maximum lifespan can occasionally reach up to 80 years'
  answer part of the token table:
    69  tmpl_03   'assistant'               
    70  tmpl_04   '\n'                      assistant_tmp_end
    71  ans_00    'Ele'                     answer_first
    72  ans_01    'phants'                  
    73  ans_02    ' typically'              
    74  ans_03    ' live'                   
    75  ans_04    ' for'                    
    76  ans_05    ' '                       
    77  ans_06    '60'                      
    78  ans_07    ' to'                     
    79  ans_08    ' '                       
    80  ans_09    '70'                      
    81  ans_10    ' years'                  
    82  ans_11    ' in'                     
    83  ans_12    ' the'                    
    84  ans_13    ' wild'                   
    85  ans_14    ','                       
    86  ans_15    ' though'                 
    87  ans_16    ' their'                  
    88  ans_17    ' maximum'                
    89  ans_18    ' lifespan'               
    90  ans_19    ' can'                    
    91  ans_20    ' occasionally'           
    92  ans_21    ' reach'                  
    93  ans_22    ' up'                     
    94  ans_23    ' to'                     
    95  ans_24    ' '                       
    96  ans_25    '80'                      
    97  ans_26    ' years'                  
    98  ans_27    '.'                       
    99  ans_28    ' Given'                  
   100  ans_29    ' the'                    
   101  ans_30    ' question'               
   102  ans_31    ' refers'                 
   103  ans_32    ' to'                     
   104  ans_33    ' a'                      
   105  ans_34    ' "'                      
   106  ans_35    't'                       
   107  ans_36    'usk'                     
   108  ans_37    'ed'                      
   109  ans_38    ' mamm'                   
   110  ans_39    'al'                      
   111  ans_40    ' with'                   
   112  ans_41    ' a'                      
   113  ans_42    ' long'                   
   114  ans_43    ' trunk'                  
   115  ans_44    ' that'                   
   116  ans_45    ' mourn'                  
   117  ans_46    's'                       
   118  ans_47    ' its'                    answer_last last
  [OK] last is the final token
  [OK] answer_last == last
  [WARN] no end token within 48 tokens: the answer may be cut off, answer_last/last sit on the cap, and 'correct' may be wrong

--- random 3560
  answer 'Four'  correct=True  capped=False  gen_ids [28070, 100257]
  [OK] unbatched generation gives the same ids
  answer part of the token table:
    71  tmpl_03   'assistant'               
    72  tmpl_04   '\n'                      assistant_tmp_end
    73  ans_00    'Four'                    answer_first answer_last
    74  ans_01    '<|endoftext|>'           last
  [OK] last is the final token
  [OK] answer_last precedes the end token

====================================================================================================
3  every probed position -> token per row
====================================================================================================
  135 positions on these 5 rows (the full run has more: longest segments)
  fixed markers (labels): {'sys_00': ['seq_begin'], 'tmpl_01': ['end_user'], 'tmpl_02': ['assistant_tmp_begin'], 'tmpl_04': ['assistant_tmp_end'], 'ans_00': ['answer_first']}
  moving markers (own positions): ['ident_last', 'answer_last', 'last']

  position       markers                row0               row1               row2               row3               row4              
  sys_00         seq_begin                0 '<|im_start|>    0 '<|im_start|>    0 '<|im_start|>    0 '<|im_start|>    0 '<|im_start|> 
  sys_01                                  1 'system'         1 'system'         1 'system'         1 'system'         1 'system'      
  sys_02                                  2 '\n'             2 '\n'             2 '\n'             2 '\n'             2 '\n'          
  sys_03                                  3 'You'            3 'You'            3 'You'            3 'You'            3 'You'         
  sys_04                                  4 ' are'           4 ' are'           4 ' are'           4 ' are'           4 ' are'        
  sys_05                                  5 ' a'             5 ' a'             5 ' a'             5 ' a'             5 ' a'          
  sys_06                                  6 ' helpful'       6 ' helpful'       6 ' helpful'       6 ' helpful'       6 ' helpful'    
  sys_07                                  7 ' function'      7 ' function'      7 ' function'      7 ' function'      7 ' function'   
  sys_08                                  8 '-c'             8 '-c'             8 '-c'             8 '-c'             8 '-c'          
  sys_09                                  9 'alling'         9 'alling'         9 'alling'         9 'alling'         9 'alling'      
  sys_10                                 10 ' AI'           10 ' AI'           10 ' AI'           10 ' AI'           10 ' AI'         
  sys_11                                 11 ' assistant'    11 ' assistant'    11 ' assistant'    11 ' assistant'    11 ' assistant'  
  sys_12                                 12 '.'             12 '.'             12 '.'             12 '.'             12 '.'           
  sys_13                                 13 ' You'          13 ' You'          13 ' You'          13 ' You'          13 ' You'        
  sys_14                                 14 ' do'           14 ' do'           14 ' do'           14 ' do'           14 ' do'         
  sys_15                                 15 ' not'          15 ' not'          15 ' not'          15 ' not'          15 ' not'        
  sys_16                                 16 ' currently'    16 ' currently'    16 ' currently'    16 ' currently'    16 ' currently'  
  sys_17                                 17 ' have'         17 ' have'         17 ' have'         17 ' have'         17 ' have'       
  sys_18                                 18 ' access'       18 ' access'       18 ' access'       18 ' access'       18 ' access'     
  sys_19                                 19 ' to'           19 ' to'           19 ' to'           19 ' to'           19 ' to'         
  sys_20                                 20 ' any'          20 ' any'          20 ' any'          20 ' any'          20 ' any'        
  sys_21                                 21 ' functions'    21 ' functions'    21 ' functions'    21 ' functions'    21 ' functions'  
  sys_22                                 22 '.'             22 '.'             22 '.'             22 '.'             22 '.'           
  sys_23                                 23 ' '             23 ' '             23 ' '             23 ' '             23 ' '           
  sys_24                                 24 '<functions>'   24 '<functions>'   24 '<functions>'   24 '<functions>'   24 '<functions>' 
  sys_25                                 25 '</functions>   25 '</functions>   25 '</functions>   25 '</functions>   25 '</functions> 
  sys_26                                 26 '<|im_end|>'    26 '<|im_end|>'    26 '<|im_end|>'    26 '<|im_end|>'    26 '<|im_end|>'  
  sys_27                                 27 '\n'            27 '\n'            27 '\n'            27 '\n'            27 '\n'          
  sys_28                                 28 '<|im_start|>   28 '<|im_start|>   28 '<|im_start|>   28 '<|im_start|>   28 '<|im_start|> 
  sys_29                                 29 'user'          29 'user'          29 'user'          29 'user'          29 'user'        
  sys_30                                 30 '\n'            30 '\n'            30 '\n'            30 '\n'            30 '\n'          
  pre_00                                 31 'What'          31 'How'           31 'How'           31 'For'           31 'How'         
  pre_01                                 32 ' number'       32 ' long'         32 ' many'         32 ' how'          32 ' many'       
  pre_02                                 33 ' of'           33 '-lived'        33 ' offspring'    33 ' many'         33 ' limbs'      
  pre_03                                 34 ' limbs'        34 ' is'           34 ' does'         34 ' years'        34 ' does'       
  pre_04                                 35 ' does'          -                  -                 35 ' does'          -               
  id_00                                  36 ' the'          35 ' the'          35 ' the'          36 ' the'          35 ' the'        
  id_01                                  37 ' mamm'         36 ' fastest'      36 ' big'          37 ' t'            36 ' long'       
  id_02                                  38 'al'            37 ' land'         37 '-foot'         38 'usk'           37 '-cl'         
  id_03                                  39 ' of'           38 ' animal'       38 'ed'            39 'ed'            38 'aw'          
  id_04                                  40 ' the'          39 ','             39 ' Australian'   40 ' mamm'         39 'ed'          
  id_05                                  41 ' Ne'           40 ' an'           40 ' mars'         41 'al'            40 ' mamm'       
  id_06                                  42 'ot'            41 ' African'      41 'up'            42 ' with'         41 'al'          
  id_07                                  43 'ropical'       42 ' cat'          42 'ial'           43 ' a'            42 ' of'         
  id_08                                  44 ' canopy'       43 ' whose'        43 ' that'         44 ' long'         43 ' the'        
  id_09                                  45 ','             44 ' spine'        44 ' boxes'        45 ' trunk'        44 ' Ne'         
  id_10                                  46 ' famed'        45 ' flex'         45 ' with'         46 ' that'         45 'ot'          
  id_11                                  47 ' for'          46 'es'            46 ' its'          47 ' mourn'        46 'ropical'     
  id_12                                  48 ' its'          47 ' so'           47 ' fore'         48 's'             47 ' canopy'     
  id_13                                  49 ' leth'         48 ' far'          48 'arms'          49 ' its'          48 ','           
  id_14                                  50 'arg'           49 ' that'         49 ' when'         50 ' dead'         49 ' famed'      
  id_15                                  51 'y'             50 ' it'           50 ' challenged'   51 ' and'          50 ' for'        
  id_16                                  52 ','             51 ' is'            -                 52 ' revis'        51 ' its'        
  id_17                                  53 ' whose'        52 ' airborne'      -                 53 'its'           52 ' leth'       
  id_18                                  54 ' sh'           53 ' for'           -                 54 ' their'        53 'arg'         
  id_19                                  55 'ag'            54 ' much'          -                 55 ' bones'        54 'y'           
  id_20                                  56 'gy'            55 ' of'            -                  -                 55 ','           
  id_21                                  57 ' coat'         56 ' each'          -                  -                 56 ' that'       
  id_22                                  58 ' grows'        57 ' stride'        -                  -                 57 ' nonetheless 
  id_23                                  59 ' algae'         -                  -                  -                 58 ' swims'      
  id_24                                  60 ' and'           -                  -                  -                 59 ' surprisingl 
  id_25                                  61 ' shelters'      -                  -                  -                 60 ' well'       
  id_26                                  62 ' mo'            -                  -                  -                  -               
  id_27                                  63 'ths'            -                  -                  -                  -               
  post_00                                64 ' have'         58 ' on'           51 ' have'         56 ' normally'     61 ' have'       
  post_01                                65 '?'             59 ' average'      52 ' in'           57 ' live'         62 '?'           
  post_02                                66 ' Answer'       60 '?'             53 ' a'            58 '?'             63 ' Answer'     
  post_03                                67 ' in'           61 ' Answer'       54 ' single'       59 ' Answer'       64 ' in'         
  post_04                                68 ' one'          62 ' with'         55 ' reproductiv   60 ' with'         65 ' one'        
  post_05                                69 ' word'         63 ' a'            56 ' event'        61 ' a'            66 ' word'       
  post_06                                70 '.'             64 ' number'       57 '?'             62 ' number'       67 '.'           
  post_07                                 -                 65 ' of'           58 ' Answer'       63 ' of'            -               
  post_08                                 -                 66 ' years'        59 ' with'         64 ' years'         -               
  post_09                                 -                 67 '.'             60 ' a'            65 '.'              -               
  post_10                                 -                  -                 61 ' number'        -                  -               
  post_11                                 -                  -                 62 ' or'            -                  -               
  post_12                                 -                  -                 63 ' a'             -                  -               
  post_13                                 -                  -                 64 ' range'         -                  -               
  post_14                                 -                  -                 65 '.'              -                  -               
  tmpl_00                                71 '<|im_end|>'    68 '<|im_end|>'    66 '<|im_end|>'    66 '<|im_end|>'    68 '<|im_end|>'  
  tmpl_01        end_user                72 '\n'            69 '\n'            67 '\n'            67 '\n'            69 '\n'          
  tmpl_02        assistant_tmp_begin     73 '<|im_start|>   70 '<|im_start|>   68 '<|im_start|>   68 '<|im_start|>   70 '<|im_start|> 
  tmpl_03                                74 'assistant'     71 'assistant'     69 'assistant'     69 'assistant'     71 'assistant'   
  tmpl_04        assistant_tmp_end       75 '\n'            72 '\n'            70 '\n'            70 '\n'            72 '\n'          
  ans_00         answer_first            76 'Four'          73 'The'           71 'The'           71 'Ele'           73 'Four'        
  ans_01                                 77 '<|endoftext|   74 ' fastest'      72 ' big'          72 'phants'        74 '<|endoftext| 
  ans_02                                  -                 75 ' land'         73 '-foot'         73 ' typically'     -               
  ans_03                                  -                 76 ' animal'       74 'ed'            74 ' live'          -               
  ans_04                                  -                 77 ' is'           75 ' Australian'   75 ' for'           -               
  ans_05                                  -                 78 ' the'          76 ' mars'         76 ' '              -               
  ans_06                                  -                 79 ' che'          77 'up'            77 '60'             -               
  ans_07                                  -                 80 'et'            78 'ial'           78 ' to'            -               
  ans_08                                  -                 81 'ah'            79 ' you'          79 ' '              -               
  ans_09                                  -                 82 '.'             80 "'re"           80 '70'             -               
  ans_10                                  -                 83 ' While'        81 ' referring'    81 ' years'         -               
  ans_11                                  -                 84 ' the'          82 ' to'           82 ' in'            -               
  ans_12                                  -                 85 ' question'     83 ' is'           83 ' the'           -               
  ans_13                                  -                 86 ' refers'       84 ' likely'       84 ' wild'          -               
  ans_14                                  -                 87 ' to'           85 ' the'          85 ','              -               
  ans_15                                  -                 88 ' a'            86 ' **'           86 ' though'        -               
  ans_16                                  -                 89 ' "'            87 'T'             87 ' their'         -               
  ans_17                                  -                 90 'A'             88 'as'            88 ' maximum'       -               
  ans_18                                  -                 91 'frican'        89 'man'           89 ' lifespan'      -               
  ans_19                                  -                 92 ' cat'          90 'ian'           90 ' can'           -               
  ans_20                                  -                 93 '"'             91 ' devil'        91 ' occasionall    -               
  ans_21                                  -                 94 ' with'         92 '**'            92 ' reach'         -               
  ans_22                                  -                 95 ' a'            93 ' (*'           93 ' up'            -               
  ans_23                                  -                 96 ' highly'       94 'S'             94 ' to'            -               
  ans_24                                  -                 97 ' flexible'     95 'arc'           95 ' '              -               
  ans_25                                  -                 98 ' spine'        96 'ophil'         96 '80'             -               
  ans_26                                  -                 99 ','             97 'us'            97 ' years'         -               
  ans_27                                  -                100 ' this'         98 ' h'            98 '.'              -               
  ans_28                                  -                101 ' description   99 'arris'         99 ' Given'         -               
  ans_29                                  -                102 ' most'        100 'ii'           100 ' the'           -               
  ans_30                                  -                103 ' closely'     101 '*'            101 ' question'      -               
  ans_31                                  -                104 ' matches'     102 '),'           102 ' refers'        -               
  ans_32                                  -                105 ' the'         103 ' though'      103 ' to'            -               
  ans_33                                  -                106 ' che'         104 ' it'          104 ' a'             -               
  ans_34                                  -                107 'et'           105 ' is'          105 ' "'             -               
  ans_35                                  -                108 'ah'           106 ' not'         106 't'              -               
  ans_36                                  -                109 ' ('           107 ' known'       107 'usk'            -               
  ans_37                                  -                110 'Ac'           108 ' for'         108 'ed'             -               
  ans_38                                  -                111 'in'           109 ' boxing'      109 ' mamm'          -               
  ans_39                                  -                112 'ony'          110 ' with'        110 'al'             -               
  ans_40                                  -                113 'x'            111 ' its'         111 ' with'          -               
  ans_41                                  -                114 ' jub'         112 ' fore'        112 ' a'             -               
  ans_42                                  -                115 'atus'         113 'arms'         113 ' long'          -               
  ans_43                                  -                116 '),'           114 '.'            114 ' trunk'         -               
  ans_44                                  -                117 ' which'       115 ' However'     115 ' that'          -               
  ans_45                                  -                118 ' can'         116 ','            116 ' mourn'         -               
  ans_46                                  -                119 ' reach'       117 ' among'       117 's'              -               
  ans_47                                  -                120 ' speeds'      118 ' mars'        118 ' its'           -               
  ident_last                             63 'ths'           57 ' stride'       50 ' challenged'   55 ' bones'        60 ' well'       
  answer_last                            76 'Four'           -                  -                  -                 73 'Four'        
  last                                   77 '<|endoftext|    -                  -                  -                 74 '<|endoftext| 
  [OK] every token of every row maps to a probed coordinate

====================================================================================================
4  collect(): stored states vs a plain unbatched forward
====================================================================================================
rank 0: forward 5/5
  A shape (135, 32, 5, 4096) = (positions, states, rows, hidden), dtype float16
  row0 relative error: median 9.85e-03  worst 3.10e-02 at pre_02 h31
  row1 relative error: median 1.05e-02  worst 5.96e-02 at ans_32 h07
  row2 relative error: median 9.98e-03  worst 3.10e-02 at pre_03 h31
  row3 relative error: median 1.04e-02  worst 2.51e-02 at id_11 h31
  row4 relative error: median 9.77e-03  worst 2.85e-02 at id_19 h31

  per state: median / worst relative error vs median distance to the neighbouring token
    h00  median 0.00e+00  worst 4.22e-08   neighbour 1.27e+00   worst ratio 30200990.0
    h01  median 3.42e-03  worst 1.13e-02   neighbour 1.16e+00   worst ratio    102.2
    h02  median 4.75e-03  worst 2.30e-02   neighbour 1.16e+00   worst ratio     50.6
    h03  median 5.72e-03  worst 2.75e-02   neighbour 1.14e+00   worst ratio     41.3
    h04  median 6.59e-03  worst 3.13e-02   neighbour 1.12e+00   worst ratio     35.7
    h05  median 7.21e-03  worst 3.94e-02   neighbour 1.06e+00   worst ratio     27.0
    h06  median 7.60e-03  worst 4.91e-02   neighbour 1.05e+00   worst ratio     21.4
    h07  median 8.03e-03  worst 5.96e-02   neighbour 1.01e+00   worst ratio     16.9
    h08  median 8.43e-03  worst 4.01e-02   neighbour 9.95e-01   worst ratio     24.8
    h09  median 8.78e-03  worst 5.04e-02   neighbour 9.85e-01   worst ratio     19.5
    h10  median 9.17e-03  worst 5.15e-02   neighbour 9.75e-01   worst ratio     18.9
    h11  median 9.28e-03  worst 4.95e-02   neighbour 9.59e-01   worst ratio     19.4
    h12  median 9.75e-03  worst 4.22e-02   neighbour 9.69e-01   worst ratio     23.0
    h13  median 1.03e-02  worst 3.97e-02   neighbour 9.86e-01   worst ratio     24.8
    h14  median 1.05e-02  worst 3.01e-02   neighbour 9.76e-01   worst ratio     32.4
    h15  median 1.07e-02  worst 2.72e-02   neighbour 9.71e-01   worst ratio     35.7
    h16  median 1.09e-02  worst 2.51e-02   neighbour 9.71e-01   worst ratio     38.6
    h17  median 1.09e-02  worst 2.27e-02   neighbour 9.89e-01   worst ratio     43.6
    h18  median 1.09e-02  worst 2.19e-02   neighbour 9.96e-01   worst ratio     45.5
    h19  median 1.09e-02  worst 2.01e-02   neighbour 9.70e-01   worst ratio     48.3
    h20  median 1.11e-02  worst 2.06e-02   neighbour 9.91e-01   worst ratio     48.1
    h21  median 1.14e-02  worst 2.11e-02   neighbour 9.98e-01   worst ratio     47.2
    h22  median 1.19e-02  worst 2.20e-02   neighbour 1.00e+00   worst ratio     45.6
    h23  median 1.22e-02  worst 2.29e-02   neighbour 1.01e+00   worst ratio     44.1
    h24  median 1.29e-02  worst 2.38e-02   neighbour 1.02e+00   worst ratio     42.9
    h25  median 1.34e-02  worst 2.47e-02   neighbour 1.03e+00   worst ratio     41.7
    h26  median 1.38e-02  worst 2.62e-02   neighbour 1.03e+00   worst ratio     39.3
    h27  median 1.44e-02  worst 2.71e-02   neighbour 1.05e+00   worst ratio     38.6
    h28  median 1.49e-02  worst 2.89e-02   neighbour 1.07e+00   worst ratio     36.9
    h29  median 1.52e-02  worst 2.98e-02   neighbour 1.08e+00   worst ratio     36.2
    h30  median 1.53e-02  worst 3.08e-02   neighbour 1.09e+00   worst ratio     35.3
    h31  median 1.54e-02  worst 3.10e-02   neighbour 1.08e+00   worst ratio     34.9
  [OK] aligned: every stored state is >10x closer to its own token than to the neighbour
rank 0: forward 5/5
  max |order A - order B| per state: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [OK] states independent of batch order and slot (bitwise)
  [OK] h[0] identical at all 40 positions where the rows share a token 
  sys_00 spread across rows per state: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [OK] sys_00 identical in every row at every state (fixed batch shape works)

====================================================================================================
5  hidden state vs residual delta at assistant_tmp_end
====================================================================================================
  [OK] d01 == h01 - h00
  [OK] d16 == h16 - h15
  [OK] d31 == h31 - h30

  layer   |h| mean   |d| mean  |d|/|h|  cos(d, h[k-1])
      0       1.51
      1       1.46       1.67    1.145          -0.582
      2       1.72       1.15    0.672          -0.156
      3       1.96       0.71    0.363           0.168
      4       2.31       1.22    0.528          -0.004
      5       2.49       1.08    0.435          -0.057
      6       3.24       1.84    0.568           0.099
      7       3.81       2.10    0.552          -0.033
      8       4.57       2.59    0.566          -0.022
      9       4.75       2.18    0.459          -0.153
     10       5.58       2.86    0.510           0.013
     11       6.65       3.36    0.507           0.045
     12       8.04       4.24    0.528           0.042
     13       8.14       3.74    0.460          -0.203
     14       9.56       5.00    0.521           0.000
     15      11.09       5.89    0.531          -0.026
     16      12.42       6.58    0.530          -0.080
     17      13.89       7.12    0.514          -0.066
     18      15.57       7.75    0.498          -0.051
     19      18.11       8.97    0.495           0.021
     20      20.11       8.28    0.411           0.024
     21      22.97       9.84    0.428           0.066
     22      24.96       9.83    0.392          -0.008
     23      27.74      10.35    0.374           0.078
     24      29.88      10.15    0.340           0.037
     25      32.20      10.58    0.329           0.052
     26      35.50      11.93    0.337           0.104
     27      38.29      12.30    0.322           0.062
     28      42.52      14.30    0.336           0.126
     29      45.95      15.39    0.335           0.051
     30      52.66      21.57    0.409           0.097
     31      59.12      23.01    0.389           0.080

====================================================================================================
6  probe mechanics on these 5 rows (train = eval = all 5; accuracy must be 1.0)
====================================================================================================
  classes ['Cheetah', 'Elephant', 'Kangaroo', 'Sloth'] -> y [3 0 2 1 3]
  X (3, 5, 4096) = (probes fitted together, rows, hidden)
  h31: ours train 1.00 | sklearn train 1.00
  [OK] h31 separates 5 rows
  d31: ours train 1.00 | sklearn train 1.00
  [OK] d31 separates 5 rows
  h00: ours train 0.40 (constant input, expected majority share 0.40)
  [OK] h00 null probe falls back to one class

====================================================================================================
7  splits on the full table (no model involved)
====================================================================================================
  19630 rows, 24 classes, n_match 5880
  ident      train  5880  dev  4290  test  4270
  frame      train  5880  dev  3432  test  3068
  slot/0     train  5880  dev  2870  test  5050
  slot/1     train  5880  dev  2850  test  5150
  slot/2     train  5880  dev  2860  test  5140
  strict/0   train  5342  dev  2296  test   280
  strict/1   train  5880  dev  2280  test   286
  strict/2   train  5334  dev  2288  test   288
  longest segments: {'sys': 31, 'pre': 13, 'id': 33, 'post': 19, 'tmpl': 5, 'ans': 48}

  id_00: coverage 1.000 (19630 rows), animals present 24
    ident      train  5880  dev  4290  test  4270
    frame      train  5880  dev  3432  test  3068
    slot/0     train  5880  dev  2870  test  5050
    slot/1     train  5880  dev  2850  test  5150
    slot/2     train  5880  dev  2860  test  5140
    strict/0   train  5342  dev  2296  test   280
    strict/1   train  5880  dev  2280  test   286
    strict/2   train  5334  dev  2288  test   288

  id_16: coverage 0.606 (11890 rows), animals present 24
    ident      train  3504  dev  2770  test  2140
    frame      train  3628  dev  2216  test  1824
    slot/0     train  3492  dev  1850  test  3020
    slot/1     train  3482  dev  1840  test  3050
    slot/2     train  3482  dev  1850  test  3050
    strict/0   train  2924  dev  1480  test   142
    strict/1   train  3712  dev  1472  test   142
    strict/2   train  2916  dev  1480  test   144

  id_32: coverage 0.005 (90 rows), animals present 1
    every condition skipped (< 240 train rows)

done. Anything marked FAIL above needs a look before the full run.
