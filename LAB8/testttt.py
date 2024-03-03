

def calculate_money(start_money, i, n_step):
    i_money = 0
    for j in range(n_step):
        i_money += start_money*(i**j)
    
    total_money =  i_money/n_step
    return total_money

print(calculate_money(100, 0.1, 12))
