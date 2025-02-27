#include <stdio.h>
#include <stdlib.h>

void keep_first_100_lines(const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        perror("Error opening file for reading");
        return;
    }

    // Temporary storage for the first 100 lines
    char *lines[100];
    int count = 0;

    for (int i = 0; i < 100; i++) {
        lines[i] = malloc(1024); // Allocate memory for each line (assuming max length 1024)
        if (!fgets(lines[i], 1024, file)) {
            free(lines[i]); // Free memory if EOF is reached early
            break;
        }
        count++;
    }
    fclose(file);

    // Open the file for writing (this clears the file)
    file = fopen(file_path, "w");
    if (!file) {
        perror("Error opening file for writing");
        for (int i = 0; i < count; i++) {
            free(lines[i]); // Free allocated memory
        }
        return;
    }

    // Write the first 100 lines back to the file
    for (int i = 0; i < count; i++) {
        fputs(lines[i], file);
        free(lines[i]); // Free allocated memory after use
    }

    fclose(file);
}

int main() {
    const char *file_path = "/users/ymy_yuan/m3/parsimon-eval/expts/fig_8/data/0/ns3-config/2/flows.txt";
    keep_first_100_lines(file_path);
    printf("The first 100 lines are retained in the file: %s\n", file_path);
    return 0;
}
