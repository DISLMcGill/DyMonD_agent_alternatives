# file1 : file with predictions
file1 = open('ten_classes_0_results.csv','r')

# file3 : testing data (each record ends with its corresponding class)
file3 = open('../Keras2C2/TestNoSLCor.csv','r')

# file4 : file to store the correct class prediction for each test
file4 = open('the_real_predictions.txt','w')

# 'Cass': 0, 'CassMN': 1, 'DB2': 2, 'HTTP': 3, 'Memcached': 4, 'MonetDB': 5, 'MySql': 6, 'PostgreSQL': 7, 'Redis': 8, 'Spark-W': 9 

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

print(correct/5643) # total: 5643 tests

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
