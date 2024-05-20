# file1 : file with predictions
file1 = open('ten_classes_0_results.csv','r')

file4 = open('/Users/izzychampion/DyMonD_Research/ResNet/ResnetClasses.txt','r')

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

correct = 0


for line1, line2 in zip(file1,file4):
    
    elements = line1.split()
    num = line2.split()
    
    maximum = -1.0
    value = 0
    
    for i in range(len(elements)):
        if float(elements[i]) > maximum:
            value = i
            maximum = float(elements[i])
    
    if(value == int(num[0].split('.')[0])):
        print("correct " + str(value) + "\n" )
        true_pos +=1
        true_neg += 9
        correct += 1
    else:
        print("no" + "guessed " + str(value) + " right " +  str(num[0].split('.')[0]) + "\n")
        false_pos +=1
        false_neg +=1
        true_neg += 8

print(correct)

print(correct/10000) # total: 5643 tests

accuracy = (true_pos + true_neg)/(true_pos + false_pos + false_neg + true_neg)
recall = true_pos/(true_pos + false_neg)
precision = true_pos/(true_pos+false_pos)

f1 = 2* (precision * recall)/( precision + recall )
print("accuracy: " + str(accuracy) )
print("recall: " + str(recall) )
print("precision: " + str(precision) )
print("f1 : " + str(f1) )

file1.close()
file4.close()
