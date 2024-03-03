#include <stdio.h>
#include <math.h>

double calculate_money(double start_money, double i, int n_step) {
    double i_money = 0;
    int j;
    for (j = 0; j < n_step; j++) {
        i_money += start_money * pow(i, j);
    }
    
    double total_money = i_money / n_step;
    return total_money;
}

int main() {
    double start_money, i;
    int n_step;

    printf("Enter start money: ");
    scanf("%lf", &start_money);

    printf("Enter interest rate (in decimal): ");
    scanf("%lf", &i);

    printf("Enter number of steps: ");
    scanf("%d", &n_step);

    printf("Total money: %f\n", calculate_money(start_money, i, n_step));
    return 0;
}
