# ------------------------------
# 🌾 Python Basics Revision File
# ------------------------------

# --- Functions ---
def add_numbers(a, b):
    return a + b

print(add_numbers(10, 5))  # ✅ prints 15

# --- Numeric data types ---
# 1. int
a = 10
# 2. float
b = 0.1
# 3. complex
c = 2 + 4j

print(a, b, c)

# --- Strings ---
name = "sujal"
print(name[0])  # prints 's'

# --- Tuples ---
t = (1, 2, 3, 4)
print(t)

# --- Lists ---
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
print(fruits)
fruits.remove("banana")
print(fruits)

# --- Dictionary Example ---
database = {
    "customer_id": 1,
    "customer_name": "Ayushman",
    "product": "orange",
    "quantity": 4,
    "price": 200
}
print(database)

# --- Machine Learning Basics ---
"""
Supervised Learning: input + output (classification, regression)
Unsupervised Learning: input only (clustering)
Reinforcement Learning: trial & error (self-driving car)
"""

# --- Class and Object Example ---
class Car:
    def start(self):
        print("Car started")

my_car = Car()
my_car.start()

# --- Another Class Example ---
class Kitchen:
    def cook(self):
        print("Cooking Maggi with water, noodles, and masala...")

my_kitchen = Kitchen()
my_kitchen.cook()

# --- Constructor Example ---
class Car2:
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color

    def show(self):
        print(f"This is a {self.color} {self.brand} car.")

my_car2 = Car2("BMW", "Black")
my_car2.show()

# --- Movie Example ---
class Movie:
    def __init__(self, movie, rating):
        self.movie = movie
        self.rating = rating

    def show(self):
        print(f"This is the movie '{self.movie}' with a rating of {self.rating}.")

my_movie = Movie("Kantara", "9")
my_movie.show()

# --- Inheritance Example ---
class Vehicle:
    def start(self):
        print("Vehicle started")

class Car3(Vehicle):
    def drive(self):
        print("Car is driving")

my_car3 = Car3()
my_car3.start()
my_car3.drive()

# --- Inheritance Example (Shopping) ---
class Shopping:
    def buy(self):
        print("Item added to cart and ready to buy")

class Payment(Shopping):
    def payment(self):
        print("Payment page is ready")

my_shopping = Shopping()
my_shopping.buy()
my_payment = Payment()
my_payment.payment()

# --- NumPy Section ---
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr + 10)
print(arr * 2)
print(np.sqrt(arr))

# --- Multi-dimensional Arrays ---
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)
print(matrix[0, 1])
print(matrix.T)

# --- Statistics Example ---
data = np.array([5, 10, 15, 20])
print("Mean:", np.mean(data))
print("Standard Deviation:", np.std(data))
print("Sum:", np.sum(data))

# --- Mean, Median, Mode for Cities ---
B = np.array([100, 150, 20, 450])
K = np.array([200, 220, 450, 300])
D = np.array([350, 300, 100, 250])

print("\nBangalore:", "Mean =", np.mean(B), "Std =", np.std(B), "Sum =", np.sum(B))
print("Kolkata:", "Mean =", np.mean(K), "Std =", np.std(K), "Sum =", np.sum(K))
print("Delhi:", "Mean =", np.mean(D), "Std =", np.std(D), "Sum =", np.sum(D))

# --- Random Numbers ---
random_nums = np.random.rand(5)
print("Random:", random_nums)
random_integers = np.random.randint(10, 50, size=5)
print("Random Integers:", random_integers)

# --- Project: Daily Temperature Analysis ---
temperature = np.array([30, 32, 29, 35, 31, 33, 28])
print("\nDaily temp:", temperature)
mean_temp = np.mean(temperature)
max_temp = np.max(temperature)
min_temp = np.min(temperature)
std_temp = np.std(temperature)

print("Average:", mean_temp, "Max:", max_temp, "Min:", min_temp, "SD:", std_temp)
hot_days = temperature > mean_temp
print("Days hotter than average:", hot_days)
cooler_week = temperature - 2
print("New Average after cooling:", np.mean(cooler_week))

# --- Project: Amazon Great Indian Festival ---
products = {
    "iPhone": {"original_price": 120000, "sale_price": 90000},
    "TV": {"original_price": 45000, "sale_price": 25000},
    "Fridge": {"original_price": 55000, "sale_price": 30000},
    "Laptop": {"original_price": 65000, "sale_price": 35000},
    "Headphones": {"original_price": 12500, "sale_price": 10000},
    "PS5": {"original_price": 37000, "sale_price": 25000}
}

print("\nAmazon Great Indian Shopping Festival\n")
for name, price in products.items():
    original = price["original_price"]
    sale = price["sale_price"]
    discount = original - sale
    discount_percent = (discount / original) * 100
    print(f"{name}: Original ₹{original}, Sale ₹{sale}, Discount ₹{discount} ({discount_percent:.2f}%)")

# --- Pandas Section ---
import pandas as pd

# Series
s = pd.Series([10, 20, 30, 40])
print("\nSeries:\n", s)

# DataFrame
data = {
    "Name": ["Ayushman", "Harsh", "Kapil", "Karan"],
    "Age": [25, 20, 18, 20],
    "Score": [91, 45, 89, 67]
}
df = pd.DataFrame(data)
print("\nDataFrame:\n", df)

# Handle Missing Values
df2 = pd.DataFrame({"Name": ["Ayushman", "Harsh", "Kapil", None],
                    "Age": [25, 20, 18, None],
                    "Score": [91, 45, 89, 67]})
print(df2.isnull())

# Statistics
df3 = pd.DataFrame({"Age": [25, 23, 22, 18], "Score": [50, 27, 37, 79]})
print(df3.describe())
print(df3["Age"].max())

# Filtering
df4 = pd.DataFrame({"Name": ["Shambhavi", "Ashutosh", "None"],
                    "Age": [18, None, 25],
                    "Score": [90, 70, 78]})
print(df4["Name"])
print(df4[df4["Score"] > 80])

# --- University Database ---
df_uni = pd.DataFrame({
    "Name": ["Ishu", "Sujal", "Sanjana", "Andy", "Anmol", "Lakshya", "Shruti", "Bhumika", "Shubham", "Chandan"],
    "Age": [20, 20, 21, 18, 20, 18, 17, 21, 19, 20],
    "ID": [806, 909, 7989, 9832, 8921, 128, 281, 347, 89023, 9023],
    "Score": [99, 90, 80, 70, 60, 50, 40, 30, 20, 10]
})
avg_score = df_uni["Score"].mean()
print("Average Score:", avg_score)
print("Students below average:\n", df_uni[df_uni["Score"] < avg_score])

# --- Supermarket Project ---
data = {
    "INVOICE ID": [101, 102, 103, 104, 105],
    "item": ["Milk", "bread", "butter", "eggs", "bread"],
    "quantity": [10, 5, 8, 7, 6],
    "price": [30, 20, 50, 10, 20],
    "city": ["delhi", "mumbai", "delhi", "kolkata", "delhi"]
}
df_market = pd.DataFrame(data)
df_market["revenue"] = df_market["quantity"] * df_market["price"]
print("\nSupermarket Data:\n", df_market)
print("Total Revenue:", df_market["revenue"].sum())

# --- Agriculture Profit Project ---
crops = ['Wheat', 'Rice', 'Corn', 'Soybean']
seed_cost = np.array([12000, 15000, 10000, 8000])
fertilizer_cost = np.array([5000, 7000, 4000, 3000])
labor_cost = np.array([10000, 12000, 8000, 7000])
maintenance_cost = np.array([2000, 2500, 1500, 1200])
yield_amount = np.array([5000, 4500, 4000, 3500])
selling_price = np.array([5.0, 6.0, 4.5, 5.5])

total_cost = seed_cost + fertilizer_cost + labor_cost + maintenance_cost
total_income = yield_amount * selling_price
profit = total_income - total_cost

df_agri = pd.DataFrame({
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
})

print("\nAgriculture Profit Analysis (1 Year)\n", df_agri)
print("\nSummary:")
print("Total Spent:", np.sum(total_cost))
print("Total Income:", np.sum(total_income))
print("Total Profit:", np.sum(profit))
