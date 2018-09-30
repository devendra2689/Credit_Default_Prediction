set term png size 600, 400
set output "knn.png"

set yrange [0:100]
TITLE="Knn Accuracy against n_neighbors"
set title TITLE
set ylabel "Accuracy"
set xlabel "n_neighbors"
# Make the x axis labels easier to read.
set xtics rotate by 45 scale 0 right

# Select histogram data
set style data histogram
set style fill solid border
set boxwidth 0.20

plot 'knn_data' using 2:xtic(1) title "Accuracy" with boxes linetype 4  
