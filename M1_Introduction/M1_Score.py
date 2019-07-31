import argparse
import wer

# create a function that calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER should include the the total errors as well as the
# separately reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#
def read_file(file):
    n=0 #total number of  sentences
    with open(file, 'r') as f:
        utterances={}
        for line in f:
            n+=1
            line =line.strip().split()
            utterances[line[-1]]=' '.join(line[:-1])   #key is name of wav file , value is transcription of this wav file
    return n ,utterances
        
def score(ref_trn=None, hyp_trn=None):
    
    num_ref_sent ,ref_dict =read_file(ref_trn)  #refernce transcription
    _ ,hyp_dict = read_file(hyp_trn)
    num_err_sents=0   #number of sentences with errors
    total_ref_w =0  #total number of reference words 
    w_errors =0 #total number of word errors
    total_deletions =0
    total_insertions=0
    total_substitutions=0
    for k in ref_dict:
        if k in hyp_dict:
            #print(k)
            num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref_dict[k], hyp_dict[k])
           # print(num_tokens, num_errors, num_deletions, num_insertions, num_substitutions)
            total_ref_w+=num_tokens #increas number of reference words
            if num_errors !=0:
                num_err_sents+=1   #increase number of error sentences
                w_errors+=num_errors #increase number of error words
                total_deletions+=num_deletions
                total_insertions+=num_insertions
                total_substitutions+=num_substitutions
    ###reports##
    WER = w_errors/total_ref_w          #word error rate
    SER = num_err_sents/num_ref_sent    #sentence error rate
    deletions_per = total_deletions/total_ref_w    #he percentage of deletions
    insertions_per = total_insertions /total_ref_w  #The percentage of insertions
    substitutions_per = total_substitutions/total_ref_w   #The percentage of substitutions
    
   
    output= """Total number of reference sentences in the test set: {} sentences \n  
               Number of sentences with an error :{} \n
               SER :{:2f} % \n 
               Total number of reference words : {} words \n 
               Total number of word errors : {} \n 
               Total number of word substitutions, insertions, and deletions : {} , {} , {}  \n
               WER :{:2f} % \n "
               percentage of substitutions, insertions, and deletions : {:2f} % ,{:2f} % , {:2f} % """.format( num_ref_sent,
                                                                                                 num_err_sents,
                                                                                                 100*SER,
                                                                                                 total_ref_w,
                                                                                                 w_errors,
                                                                                                 total_substitutions,
                                                                                                total_insertions,
                                                                                                total_deletions,
                                                                                                100*WER,
                                                                                                100*substitutions_per,
                                                                                                100*insertions_per ,
                                                                                                100*deletions_per)
    print(output)

    return WER ,SER


if __name__=='__main__':
    """parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")"""
        
    reftrn=input('rference :')
    hyptrn =input('hypotheses:')
    score(ref_trn=reftrn, hyp_trn=hyptrn)
    #score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)
