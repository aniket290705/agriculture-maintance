function:
def add_numbers(a,b):
return a+b
print(add_numbers(10,5) 
numeric data stores
1.int:10
2.float:0.1
3.complex:2+4x
strings : alphabets data type
name="sujal"
print(name[0])
turples:
t=(1,2,3,4)
lists:
fruits =["apple","banana",cherry"]
fruits.append ("orange")
printf(fruits)
fruits.remove("banana")


Q.create a list of 3 customer ID,3 customer ID, 3 customer name,3 products quality per product.final price
print statement at last

database={
"customer_id":001,
"customer_name":"ayuhman",
"product":"orange",
"quality":4,
"price":200
}
print(database)



04/11/25
#ML
Surprvised:input+output:classification(catograziation)
unsurprised :input:regresation
reinforcement :NO input+No output (ex: self driving car)
(trail+error)
ex:spam detection
house price prediction

*supervised example
stock market
image classification
medical diagosins

*unsupervised example 
temperature prediction
customer prediction
~~~Netflix
~~~~amazon 





*surpvised prediction

*simple linear equation
unknown variable x,y
x---valve
ex house prediction'----size

2. multiple linear regression
n unknown variable
x=ay1+by2+c 

ex:house price prediction

~~ size
~~ locality


*GEN ai 
~~ ai?:simulation of humian thinking
general ai
super intelligent (theoretical)
3.narrow( specific task)
*gen ai: video creation , text, speech, application
*prompt :nlp: nature language processing


how it works
1.transformer model


why: faster output


components of gen ai
neural networks: sophisticated models+ process information through multiple layer.
collection of datasets: texts, images, speech or media
3.hardware :dedicated cpu+ gpu (computation)
4. fine tunning +adjustment : desirable outputs matching human intelligence


class ~~ blueprint/template
class car:
def start(self):
printf("car started")
#object creation
my_car = car()
my_car.start()

Q.class:kitchen:object:cook your favourite dish 



class Kitchen:
def cook(self):
print("Cooking Maggi with water, noodles, and masala... ")
my_kitchen = Kitchen()
my_kitchen.cook()
#constructor :

class Car: 
def_init_(self, brand, color):
self.brand = brand
self.color = color 
def show(self):
print(f "This is a self{self.color}{self.brand} car.")
#object Creation
my_car = Car("BMW","black")
my_car.show()
Q : fav movie name and rating given
SOLUTION : 
class Movie:
    def _init_(self, movie, rating):
        self.movie = movie
        self.rating = rating

    def show(self):
        print(f"This is the movie '{self.movie}' with a rating of {self.rating}.")

my_movie = Movie("Kantara", "9")
my_movie.show()

#object creation

def__init__(self,name,age):
self.name=name
self.age=age

student1=student ("vansh",18)
student2=student("pupii",18)

print(student1.name)
print(student2.name)

#inheritance

class vehicle:
def start (self):
print("vehicle started")
class car (vehicle):
def drive(self):
print("car is driving")
my_car = car()
my_car.start()
my_car.drive()

Q. parent class:shopping cart--shopping cart-- buy child class: payment page-- payment

sol~~~class Shopping :
    def buy(self):
       print("Item added to cart and ready to buy")

class Payment(Shopping):
    def payment(self):
        print("Payment page is ready")

my_Shopping = Shopping()
my_Shopping.buy()
my_payment = Payment()
my_payment.payment() 

python 
1. functions
2. data types
~~ string
~~integer
~~tuples
3.lists
4. class
5. object creation
6. inheritance


1. numpy
2.pandas

numerical python
~~ mathematical operations
n- dimentional array
    m*n
~~~ faster 




-----pip install (library name) help to download other resources ex photo editing or etc

#Array opererations
import numpy as np
arr =np.array([1,2,3,4,5])
print(arr+10)#adds 10 to each elements
print (arr*2)
print(np.sqrt(arr))

#multi dimensional data handling
import numpy as np
matrix = np.array([[1,2,3],[4,5,6]])

print(matrix.shape)#order
print(matrix[0,1])
print(matrix.T)

#buyer , area ,income of per person
#mathematical,statistical and linear algebra
import numpy as np
data =np.array ([5,10,15,20])
print("mean",np.mean(data))
print("standard deviation:",np.std(data))
print("sum:",np.sum(data))


Q. Find out the mean, median and mode of bangalore, Kolkata, delhi and give me your inference

bangalore: [100, 150, 20,450]

Kolkata: [200,220,450,300]

Delhi: [350,300,100,250]

import numpy as np

B = np.array([100,150,20,450])
K = np.array([200,220,450,300])
D = np.array([350,300,100,250])

print("Mean" ,np.mean(B))
print("standard Deviation:",np.std(B))
print("sum:",np.sum(B))

print("Mean" ,np.mean(K))
print("standard Deviation:",np.std(K))
print("sum:",np.sum(K))

print("Mean" ,np.mean(D))
print("standard Deviation:",np.std(D))
print("sum:",np.sum(D))



#random number generation 

import numpy as np

# generate nums = np.random.rand(5)
printf("Random:",random_nums)

# random integers in a range
print(np.random.randint(10,50,size=5))


sol----
import numpy as np


random_nums = np.random.rand(5)
print("Random:", random_nums)

random_integers = np.random.randint(10, 50, size=5)
print("Random Integers:", random_integers)



project
analyzing daily temperature data
requirement:
average temp,max,min temp
days above average
temperature variation

sol....
import numpy as np


temperature = np.array([30, 32, 29, 35, 31, 33, 28])


print("Daily temp:\n", temperature)


mean_temp = np.mean(temperature)
max_temp = np.max(temperature)
min_temp = np.min(temperature)
std_temp = np.std(temperature)


print("\nAverage Temperature:", mean_temp)
print("Max Temperature:", max_temp)
print("Min Temperature:", min_temp)
print("Standard Deviation:", std_temp)


hot_days = temperature > mean_temp
print("\nDays hotter than average (True = Hotter):\n", hot_days)

compare average before and after 

sol~~~
import numpy as np


temperature = np.array([30, 32, 29, 35, 31, 33, 28])


print("Daily temp:\n", temperature)


mean_temp = np.mean(temperature)
max_temp = np.max(temperature)
min_temp = np.min(temperature)
std_temp = np.std(temperature)


print("\nAverage Temperature:", mean_temp)
print("Max Temperature:", max_temp)
print("Min Temperature:", min_temp)
print("Standard Deviation:", std_temp)


hot_days = temperature > mean_temp
cooler_week = temperature - 2
print("\nNew Average after cooling:", np.mean(cooler_week))
PROJECT : AMAZON: GREAT INDIAN SHOPPING FESTIVAL

REQUIREMENTS
1.dataset:choose 6 product --- enter prices
2. price during sale
3. simulate customer discusion making(hint: customer most likely to choose products whose price < 45,000)

#product lsit
1.iphone:1,20,00--after sale --- 90,000
2.tv :45,000--after sale ---25,000
3.fridge:55,000--after sale ---30,000
4.laptop:65,000--after sale ---35,000
5.headphones:12,500--after sale ---10,000
6.ps5:37,000--after sale ---25,000

sol---
products = {
    "iPhone": {"original_price": 120000, "sale_price": 90000},
    "TV": {"original_price": 45000, "sale_price": 25000},
    "Fridge": {"original_price": 55000, "sale_price": 30000},
    "Laptop": {"original_price": 65000, "sale_price": 35000},
    "Headphones": {"original_price": 12500, "sale_price": 10000},
    "PS5": {"original_price": 37000, "sale_price": 25000}
}

print("Amazon Great Indian Shopping Festival\n")
print("Product Details with Discounts:\n")

for name, price in products.items():
    original = price["original_price"]
    sale = price["sale_price"]
    discount = original - sale
    discount_percent = (discount / original) * 10020
    print(f"{name}: Original ₹{original}, Sale ₹{sale}, Discount ₹{discount} ({discount_percent:.2f}%)")


#PANDAS

⦁	panel data
⦁	excel + SQL + python

1.series:1D:column
2.data frame :2D

import pandas as pd

#create a series 
s = pd.Series([10,20,30,40])
print("Series:\n",s)

#create a dataFrame
data={"Name":["Ayushman","harsh","Kapil","karan"],"Age":[25,20,18,20],"score":[91,45,89,67]}

#Data cleaning and prep
df=pd.DataFrame({"Name":["Ayushman","harsh","Kapil",None],"Age":[25,20,18,None],"Score":[91,45,89,67]})

#Handle missing values
print(df.isnull())

import pandas as pd

df = pd.DataFrame({"Age":[25,23,22,18],"Score":[50,27,37,79]})
print(df.describe())
#print(df["Score"].min())
print(df["Age"].max()
df = pd.DataFrame ({ "Name":["Shambhavi","Ashutosh","None",],"Age":[18,None,25],"Score":[90,70,78]}) 
#filtering = column
print(df["Name"])
#sorting
print(df[df["Score"]>80])

Q.Create university Database : 10 students
Student name, age, registration id and final score
identify the avg score of the class
identify students less than average score

SOLUTION-:
df = pd. DataFrame ({"Name": ["Ishu","Sujal","Sanjana","Andy","Anmol","Lakshya","Shruti","Bhumika","Shubham","Chandan"],"Age":[20,20,21,18,20,18,17,21,19,20],"ID":[806,909,7989,9832,8921,128,281,347,89023,9023],"Score":[99,90,80,70,60,50,40,30,20,10]})
print(df["Score"].mean())
print(df[df["Score"]<60])


# Grouping and Aggregation

import pandas as pd


df= pd.DataFrame({"Department": ["IT", "HR", "Finance"), "Salary": [20000, 30000, 40000]})

# Group

grouped = df.groupby("Department") ["Salary"].mean()

# Aggregation

print(grouped.mean())

sol--- 30000.0

#Read data from CSV

from google.collab import files
 uploaded = files.upload()



sol------import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame using pandas
df = pd.DataFrame({"year": [2020, 2021, 2022, 2023], "Sales": [1000, 2000, 3000, 4000]})

# Plotting the bar chart
df.plot(x="year", y="Sales", kind="bar")

# Display the plot
plt.show()

sol----

#import pandas as pd

# Data tranformation and feature engineering 
df = pd.DataFrame({"salary":[50000,60000,70000],"Bonus":[5000,6000,7000]})

#Creating new column
df["Total"]=df["salary"]+df["Bonus"]
print(df)


Q. project : supermarket sales analysis

Goal:
1. load data into frame
2.find total revenue
3. identify top sellng items
4. calculate average sales per city
5. Handle missing data

Data
INVOICE ID:101,102,103,104,105
item : Milk,bread,butter,eggs,bread
quantity:10,5,8,none,6 ~~~~relace none valve with a valve of your choice
price:30,20,50,10,20
city:delhi,mumbai,delhi,kolkata,

sol~~~
import pandas as pd
import numpy as np

data = {
    "INVOICE ID": [101, 102, 103, 104, 105],
    "item": ["Milk", "bread", "butter", "eggs", "bread"],
    "quantity": [10, 5, 8, 7, 6],
    "price": [30, 20, 50, 10, 20],
    "city": ["delhi", "mumbai", "delhi", "kolkata", "delhi"]
}

df = pd.DataFrame(data)

df["revenue"] = df["quantity"] * df["price"]
total_revenue = df["revenue"].sum()

print(df)
 
first session
 1. pickup your own project
*involve everything numpy and PANDAS

second session

* Display on steamlit 
1. what is
2. how
3.inputs
4.outputs
5.funtionlists





import numpy as np
import pandas as pd

# Crop names
crops = ['Wheat', 'Rice', 'Corn', 'Soybean']

# Costs per crop (you can edit these)
seed_cost = np.array([12000, 15000, 10000, 8000])
fertilizer_cost = np.array([5000, 7000, 4000, 3000])
labor_cost = np.array([10000, 12000, 8000, 7000])
maintenance_cost = np.array([2000, 2500, 1500, 1200])

# Yield (in kg or tons)
yield_amount = np.array([5000, 4500, 4000, 3500])

# Selling price per unit
selling_price = np.array([5.0, 6.0, 4.5, 5.5])

# Calculate totals using NumPy
total_cost = seed_cost + fertilizer_cost + labor_cost + maintenance_cost
total_income = yield_amount * selling_price
profit = total_income - total_cost

# Create DataFrame
data = {
    'Crop': crops,
    'Seed Cost': seed_cost,
    'Fertilizer Cost': fertilizer_cost,
    'Labor Cost': labor_cost,
    'Maintenance Cost': maintenance_cost,
    'Total Cost': total_cost,
    'Yield (kg)': yield_amount,
    'Selling Price': selling_price,
    'Total Income': total_income,
    'Profit': profit
}

df = pd.DataFrame(data)

# Summary
total_spent = np.sum(total_cost)
total_income_all = np.sum(total_income)
total_profit = np.sum(profit)

# Display
print(" Agriculture Profit Analysis (1 Year)\n")
print(df)
print("\n Yearly Summary:")
print(f"Total Spent: ₹{total_spent:,.2f}")
print(f"Total Income: ₹{total_income_all:,.2f}")
print(f"Total Profit: ₹{total_profit:,.2f}")















 








   




