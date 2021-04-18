




def delta(a,b):
    return a==b

# input should be two lists of nts
# where each int indicates a sentence and each value
# of each int indicates the segment to which the sentence
# belongs
# k is a window setting
# this function might have to be used for words in this setting
def get_Pk_error(ref,hyp,k):
    score = 0
    for i in range(len(ref)-k):
        score += delta(ref[i],ref[i+k]) != delta(hyp[i],hyp[i+k])
    return score/(len(ref)-k)
    

