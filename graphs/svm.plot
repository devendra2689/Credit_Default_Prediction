set term png size 600, 400
set output "svm.png"

set yrange [0:100]
TITLE="SVM Models Accuracy Comparision"
set title TITLE
set ylabel "Accuracy"
set xlabel "Kernals"
# Make the x axis labels easier to read.
set xtics rotate by 45 scale 0 right

# Select histogram data
set style data histogram
set style fill solid border
set boxwidth 0.20

plot 'svm_data' using 2:xtic(1) title "Accuracy" with boxes linetype 4  
