file = input('enter results file : ')
file1 = open(file, 'r')
file3 = open('TestNoSLCor.csv','r')

file4 = open('the_real_predictions.txt','w')

for line in file3:
    if line[-3] == 's':
        file4.write("0\n")
    elif line[-2] == 'N':
        file4.write("1\n")
    elif line[-2] == '2':
        file4.write("2\n")
    elif line[-2] == 'P':
        file4.write("3\n")
    elif line[-2] == 'd':
        file4.write("4\n")
    elif line[-2] == 'B':
        file4.write("5\n")
    elif line[-2] == 'l':
        file4.write("6\n")
    elif line[-2] == 'L':
        file4.write("7\n")
    elif line[-3] == 'i':
        file4.write("8\n")
    elif line[-2] == 'W':
        file4.write("9\n")

file4.close()
file4 = open('the_real_predictions.txt','r')

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

correct = 0

# 5643 tests

for line in file1:
    predicts = line

p = line.split()

i = 0
correct = 0
incorrect = 0
for line2 in file4:
    
    pairs = {p[i]:0,p[i+1]:1,p[i+2]:2,p[i+3]:3,p[i+4]:4,p[i+5]:5,p[i+6]:6,p[i+7]:7,p[i+8]:8}
    maximum = max(p[i],p[i+1],p[i+2],p[i+3],p[i+4],p[i+5],p[i+6],p[i+7],p[i+8]) 
    value = pairs[maximum]
    
    if (int(line2) == int(value)):
        #print("right: " + str(line2) + "pred" + str(value))
        correct+=1
        true_pos +=1
        true_neg+=9
    else:
        #print("right: " + str(line2) + "pred" + str(value))
        incorrect+=1
        false_pos +=1
        true_neg +=8
        false_neg +=1
    i+=9
print("number correct: " + str(correct))
print("percent correct: " + str(correct/5643 *100) )
print("number incorrect: " + str(incorrect))

accuracy = (true_pos + true_neg)/(true_pos + false_pos + false_neg + true_neg)
recall = true_pos/(true_pos + false_neg)
precision = true_pos/(true_pos+false_pos)

f1 = 2* (precision * recall)/( precision + recall )
print("accuracy: " + str(accuracy) )
print("recall: " + str(recall) )
print("precision: " + str(precision) )
print("f1 : " + str(f1) )

file1.close()
file3.close()
file4.close()
