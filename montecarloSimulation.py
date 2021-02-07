import random
import matplotlib
import matplotlib.pyplot as plt

#This is an example of MCMC simulation
def rollDice():
	roll = random.randint(1,100)
	if roll == 100:
		#'roll was 100'
		return False
	elif roll <= 50:
		#'roll was 1-50'
		return False
	elif 100 > roll > 50:
		#'roll was 51-99. You win'
		return True

def simple_bettor(funds, initial_wager, wager_count):
	value = funds
	wager = initial_wager
	wX = []
	vY = []	
	current_Wager = 1

	while current_Wager < wager_count:
		if rollDice():
			value += wager
			wX.append(current_Wager)
			vY.append(value)
		else:
			value -= wager
			wX.append(current_Wager)
			vY.append(value)	
		current_Wager +=1

	if value < 0:
		value = 'Broke'
	elif value > 0:
		value = 'Win'
	elif value == 0:
		value = 'Equal'
	
	plt.plot(wX, vY) # plot the distribution of data

x = 0
while  x < 100:
	simple_bettor(10000,100,100000)
	x += 1

plt.ylabel('Account Value')
plt.xlabel('Wager Count')
plt.show()
