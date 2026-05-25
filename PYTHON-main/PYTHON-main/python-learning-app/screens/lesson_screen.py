from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.metrics import dp
from kivy.app import App
from kivy.graphics import Color, RoundedRectangle
from kivy.clock import Clock

# ─────────────────────────────────────────────
#  FULL PYTHON COURSE  (theory + practice pairs)
# ─────────────────────────────────────────────
LESSONS = [
    # ── 1 ──────────────────────────────────────
    {
        "title": "1. Variables & Data Types",
        "theory": """\
WHAT IS A VARIABLE?
━━━━━━━━━━━━━━━━━━
A variable is a named container that holds a value in memory.
Python is dynamically typed — you never need to declare a type.

BUILT-IN TYPES
━━━━━━━━━━━━━━
  int    – whole numbers         x = 10
  float  – decimal numbers       pi = 3.14159
  str    – text (use '' or "")   name = "Alice"
  bool   – True or False         flag = True
  None   – absence of value      result = None

TYPE CHECKING
━━━━━━━━━━━━
Use type() to inspect a variable:
  type(42)      → <class 'int'>
  type(3.14)    → <class 'float'>
  type("hi")    → <class 'str'>

TYPE CONVERSION
━━━━━━━━━━━━━━
  int("5")    → 5
  float(3)    → 3.0
  str(100)    → "100"
  bool(0)     → False   (0, "", [], None are falsy)

NAMING RULES
━━━━━━━━━━━━
• Use snake_case: my_variable ✓
• Must start with a letter or _
• Cannot use keywords (if, for, while …)
• Case-sensitive: Name ≠ name
""",
        "practice": """\
# ── Variables & Data Types ──────────────────

# Assigning different types
age = 25
height = 5.9
name = "Bob"
is_student = True
nothing = None

print(age, type(age))
print(height, type(height))
print(name, type(name))
print(is_student, type(is_student))
print(nothing, type(nothing))

# ── Type Conversion ──────────────────────────
text_number = "42"
converted = int(text_number)
print(converted + 8)        # 50

print(float("3.14"))        # 3.14
print(str(100) + " items")  # 100 items

# ── Falsy values ─────────────────────────────
for val in [0, "", [], None, False, 0.0]:
    print(f"bool({val!r}) = {bool(val)}")

# ── Multiple assignment ───────────────────────
a, b, c = 1, 2, 3
print(a, b, c)

x = y = z = 0
print(x, y, z)
""",
    },
    # ── 2 ──────────────────────────────────────
    {
        "title": "2. Strings",
        "theory": """\
STRINGS IN PYTHON
━━━━━━━━━━━━━━━━━
Strings are immutable sequences of characters.
Use single quotes ' or double quotes " — both work.
For multi-line text use triple quotes \"\"\"...\"\"\".

COMMON OPERATIONS
━━━━━━━━━━━━━━━━━
  s = "Hello, World!"
  len(s)          → 13         (length)
  s.upper()       → "HELLO, WORLD!"
  s.lower()       → "hello, world!"
  s.strip()                    (remove leading/trailing spaces)
  s.replace("World","Python")  → "Hello, Python!"
  s.split(", ")   → ["Hello", "World!"]
  ", ".join(["a","b","c"])  → "a, b, c"

INDEXING & SLICING
━━━━━━━━━━━━━━━━━━
  s[0]     → 'H'          (first character)
  s[-1]    → '!'          (last character)
  s[0:5]   → 'Hello'      (slice: start inclusive, end exclusive)
  s[::2]   → every 2nd character
  s[::-1]  → reversed string

F-STRINGS  (Python 3.6+, preferred)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  name = "Alice"
  age  = 30
  f"My name is {name} and I am {age} years old."
  f"2 + 2 = {2 + 2}"          # expressions work too
  f"Pi ≈ {3.14159:.2f}"       # format specifiers

ESCAPE SEQUENCES
━━━━━━━━━━━━━━━━
  \\n   newline       \\t   tab
  \\'   single quote  \\\\ backslash
""",
        "practice": """\
# ── String Operations ───────────────────────

s = "  Hello, Python!  "
print(s.strip())               # Hello, Python!
print(s.strip().upper())       # HELLO, PYTHON!
print(s.strip().replace("Python", "World"))

# ── Slicing ──────────────────────────────────
text = "abcdefgh"
print(text[2:5])     # cde
print(text[::-1])    # hgfedcba  (reversed)
print(text[::2])     # aceg

# ── Splitting & joining ───────────────────────
csv = "Alice,Bob,Charlie,Diana"
names = csv.split(",")
print(names)
print(" | ".join(names))

# ── F-strings ────────────────────────────────
name  = "Aditya"
score = 87.5
print(f"Student: {name}")
print(f"Score:   {score:.1f}%")
print(f"Grade:   {'A' if score >= 90 else 'B' if score >= 75 else 'C'}")

# ── Useful checks ────────────────────────────
email = "user@example.com"
print(email.startswith("user"))   # True
print(email.endswith(".com"))     # True
print("@" in email)               # True
print(email.count("."))           # 1
""",
    },
    # ── 3 ──────────────────────────────────────
    {
        "title": "3. Control Flow (if/elif/else)",
        "theory": """\
MAKING DECISIONS
━━━━━━━━━━━━━━━━
Python uses if / elif / else to branch logic.
Indentation (4 spaces) defines the block — no braces needed.

SYNTAX
━━━━━━
  if condition:
      # block runs when condition is True
  elif another_condition:
      # checked only if first was False
  else:
      # runs when none of the above matched

COMPARISON OPERATORS
━━━━━━━━━━━━━━━━━━━━
  ==   equal to           !=   not equal
  <    less than          >    greater than
  <=   less or equal      >=   greater or equal

LOGICAL OPERATORS
━━━━━━━━━━━━━━━━━
  and   both must be True
  or    at least one must be True
  not   inverts the boolean

TRUTHINESS
━━━━━━━━━━
Falsy: 0, 0.0, "", [], {}, None, False
Everything else is truthy.

TERNARY (one-liner)
━━━━━━━━━━━━━━━━━━━
  label = "even" if n % 2 == 0 else "odd"

MATCH STATEMENT  (Python 3.10+)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  match command:
      case "quit":   ...
      case "help":   ...
      case _:        ...    # default
""",
        "practice": """\
# ── if / elif / else ─────────────────────────

score = 73

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Score {score} → Grade {grade}")

# ── Logical operators ─────────────────────────
age = 20
has_id = True

if age >= 18 and has_id:
    print("Entry allowed")
else:
    print("Entry denied")

# ── Ternary expression ────────────────────────
n = 7
parity = "even" if n % 2 == 0 else "odd"
print(f"{n} is {parity}")

# ── Nested conditions ─────────────────────────
temp = 28
humid = 80

if temp > 35:
    print("Very hot")
elif temp > 25:
    if humid > 75:
        print("Hot and humid")
    else:
        print("Warm")
else:
    print("Pleasant")

# ── FizzBuzz (classic) ────────────────────────
for i in range(1, 21):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
""",
    },
    # ── 4 ──────────────────────────────────────
    {
        "title": "4. Loops (for & while)",
        "theory": """\
FOR LOOPS
━━━━━━━━━
Iterate over any iterable (list, string, range, dict…):

  for item in collection:
      # do something with item

  range(stop)           → 0, 1, …, stop-1
  range(start, stop)    → start to stop-1
  range(start, stop, step)

  enumerate() gives index AND value:
    for i, val in enumerate(my_list):

WHILE LOOPS
━━━━━━━━━━━
Run as long as condition is True:

  while condition:
      # body
      # make sure condition eventually becomes False!

  Use break  to exit immediately.
  Use continue to skip the rest of the current iteration.
  Use pass   as a placeholder (no-op).

FOR … ELSE / WHILE … ELSE
━━━━━━━━━━━━━━━━━━━━━━━━━
The else block runs only if the loop completed
without hitting a break.

NESTED LOOPS
━━━━━━━━━━━━
Outer loop runs once for every full inner loop cycle.
Watch out: O(n²) complexity for large inputs.

LIST COMPREHENSION (compact for-loop)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  squares = [x**2 for x in range(10)]
  evens   = [x for x in range(20) if x % 2 == 0]
""",
        "practice": """\
# ── for loop with range ───────────────────────
for i in range(1, 6):
    print(f"  {i}: {'*' * i}")

# ── enumerate ────────────────────────────────
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits, start=1):
    print(f"{idx}. {fruit}")

# ── while loop ───────────────────────────────
n = 1
while n <= 16:
    print(n, end=" ")
    n *= 2
print()

# ── break & continue ─────────────────────────
for i in range(10):
    if i == 7:
        break           # stop at 7
    if i % 2 == 0:
        continue        # skip even numbers
    print(i, end=" ")   # prints: 1 3 5
print()

# ── Nested loops: multiplication table ───────
for r in range(1, 4):
    for c in range(1, 4):
        print(f"{r*c:3}", end="")
    print()

# ── List comprehension ────────────────────────
squares = [x**2 for x in range(1, 11)]
print(squares)

evens = [x for x in range(20) if x % 2 == 0]
print(evens)

# ── Sum with while ────────────────────────────
total = 0
num   = 1
while num <= 100:
    total += num
    num += 1
print(f"Sum 1-100 = {total}")   # 5050
""",
    },
    # ── 5 ──────────────────────────────────────
    {
        "title": "5. Functions",
        "theory": """\
DEFINING FUNCTIONS
━━━━━━━━━━━━━━━━━━
  def function_name(parameters):
      \"\"\"Optional docstring.\"\"\"
      # body
      return value    # optional; returns None if omitted

PARAMETERS & ARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━
  Positional:   def add(a, b): ...
  Default:      def greet(name="World"): ...
  *args:        variable number of positional args → tuple
  **kwargs:     variable number of keyword args   → dict
  Keyword-only: def f(*, key): ...  (must be passed by name)

RETURN VALUES
━━━━━━━━━━━━━
A function can return any value, or multiple (as a tuple):
  def min_max(lst):
      return min(lst), max(lst)

  lo, hi = min_max([3, 1, 4, 1, 5])

SCOPE  (LEGB Rule)
━━━━━━━━━━━━━━━━━━
Variables are looked up in this order:
  Local → Enclosing → Global → Built-in

Use global x  to modify a global inside a function.
Use nonlocal x to modify an enclosing variable.

LAMBDA (anonymous functions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  double = lambda x: x * 2
  square = lambda x: x ** 2

Useful with sorted(), map(), filter():
  nums.sort(key=lambda x: -x)   # sort descending

DOCSTRINGS
━━━━━━━━━━
  def area(r):
      \"\"\"Return the area of a circle with radius r.\"\"\"
      return 3.14159 * r * r
  help(area)   # shows the docstring
""",
        "practice": """\
# ── Basic function ────────────────────────────
def greet(name="World"):
    \"\"\"Return a greeting string.\"\"\"
    return f"Hello, {name}!"

print(greet())
print(greet("Aditya"))

# ── Multiple return values ────────────────────
def stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

data = [4, 7, 2, 9, 1, 5]
lo, hi, avg = stats(data)
print(f"min={lo}  max={hi}  avg={avg:.2f}")

# ── *args ─────────────────────────────────────
def total(*args):
    return sum(args)

print(total(1, 2, 3, 4, 5))    # 15

# ── **kwargs ──────────────────────────────────
def profile(**kwargs):
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

profile(name="Bob", age=28, city="Bengaluru")

# ── Lambda ────────────────────────────────────
square = lambda x: x ** 2
print(list(map(square, range(1, 6))))   # [1, 4, 9, 16, 25]

words = ["banana", "apple", "cherry", "date"]
words.sort(key=lambda w: len(w))
print(words)   # sorted by length

# ── Recursive function ────────────────────────
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

for i in range(1, 8):
    print(f"{i}! = {factorial(i)}")
""",
    },
    # ── 6 ──────────────────────────────────────
    {
        "title": "6. Lists & Tuples",
        "theory": """\
LISTS
━━━━━
Mutable, ordered collection of any type:
  nums = [1, 2, 3]
  mixed = [1, "hi", True, 3.14]
  nested = [[1, 2], [3, 4]]

KEY METHODS
━━━━━━━━━━━
  append(x)      add to end
  insert(i, x)   insert at index i
  remove(x)      remove first occurrence of x
  pop(i=-1)      remove & return element at i
  index(x)       find first index of x
  count(x)       count occurrences of x
  sort()         sort in place  (sorted() returns new list)
  reverse()      reverse in place
  extend(iter)   append all from iterable
  copy()         shallow copy

SLICING
━━━━━━━
  lst[start:stop:step]   (same as strings)
  lst[::-1]              reversed copy

LIST COMPREHENSION
━━━━━━━━━━━━━━━━━━
  [expr for item in iterable if condition]

TUPLES
━━━━━━
Immutable ordered collection:
  point   = (10, 20)
  rgb     = (255, 0, 128)
  single  = (42,)      ← trailing comma needed for length-1

Tuples are faster than lists and can be dict keys.
Unpacking:  x, y = point

WHEN TO USE WHICH
━━━━━━━━━━━━━━━━━
  List  → mutable collection that will change
  Tuple → fixed record (coordinates, DB row, RGB…)
""",
        "practice": """\
# ── List basics ───────────────────────────────
fruits = ["banana", "apple", "cherry"]
fruits.append("mango")
fruits.insert(1, "blueberry")
print(fruits)

fruits.remove("apple")
popped = fruits.pop()
print(f"Removed: {popped}, Remaining: {fruits}")

# ── Sorting ───────────────────────────────────
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(nums))          # new sorted list
nums.sort(reverse=True)      # in-place descending
print(nums)

# ── Slicing ───────────────────────────────────
lst = list(range(10))
print(lst[2:7])       # [2, 3, 4, 5, 6]
print(lst[::3])       # every 3rd: [0, 3, 6, 9]
print(lst[::-1])      # reversed

# ── Comprehensions ────────────────────────────
squares  = [x**2 for x in range(1, 11)]
print(squares)

even_sq  = [x**2 for x in range(1, 11) if x % 2 == 0]
print(even_sq)

matrix = [[r*c for c in range(1,4)] for r in range(1,4)]
for row in matrix:
    print(row)

# ── Tuples ───────────────────────────────────
point = (3, 7)
x, y  = point
print(f"x={x}  y={y}")

# Swap without a temp variable:
a, b = 10, 20
a, b = b, a
print(a, b)

# Tuple as dict key:
grid = {(0,0): "start", (3,3): "end"}
print(grid[(0,0)])
""",
    },
    # ── 7 ──────────────────────────────────────
    {
        "title": "7. Dictionaries & Sets",
        "theory": """\
DICTIONARIES
━━━━━━━━━━━━
Key-value store. Keys must be hashable (str, int, tuple).
Insertion order is preserved (Python 3.7+).

  person = {"name": "Alice", "age": 30}

ACCESS & MODIFICATION
━━━━━━━━━━━━━━━━━━━━━
  person["name"]              → "Alice"
  person.get("job", "N/A")    → "N/A"  (safe, no KeyError)
  person["city"] = "Delhi"    (add/update)
  del person["age"]
  person.pop("city", None)    (remove safely)

ITERATION
━━━━━━━━━
  for key in d:              (keys only)
  for val in d.values():
  for key, val in d.items(): (most common)

USEFUL METHODS
━━━━━━━━━━━━━━
  d.keys()    d.values()    d.items()
  d.update(other_dict)
  d.copy()
  dict.fromkeys(keys, default)

DICT COMPREHENSION
━━━━━━━━━━━━━━━━━━
  {k: v for k, v in pairs if condition}

SETS
━━━━
Unordered collection of unique hashable values.
  s = {1, 2, 3}   or   s = set([1, 2, 2, 3])  → {1, 2, 3}

OPERATIONS
━━━━━━━━━━
  s.add(x)     s.remove(x)   s.discard(x)
  A | B   union            A & B   intersection
  A - B   difference       A ^ B   symmetric difference
  A <= B  subset           A >= B  superset
""",
        "practice": """\
# ── Dictionary basics ────────────────────────
student = {
    "name": "Priya",
    "age": 21,
    "marks": [85, 92, 78, 95]
}

print(student["name"])
print(student.get("grade", "Not assigned"))

student["grade"] = "A"
student["marks"].append(88)
print(student)

# ── Iteration ─────────────────────────────────
for key, value in student.items():
    print(f"  {key:8} : {value}")

# ── Dict comprehension ────────────────────────
words = ["apple", "bat", "cherry", "do"]
lengths = {w: len(w) for w in words}
print(lengths)

# Invert a dict
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)

# ── Word frequency counter ────────────────────
sentence = "the cat sat on the mat the cat"
freq = {}
for word in sentence.split():
    freq[word] = freq.get(word, 0) + 1
print(freq)

# ── Sets ──────────────────────────────────────
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

print("Union:       ", A | B)
print("Intersection:", A & B)
print("Difference:  ", A - B)
print("Sym Diff:    ", A ^ B)

# Remove duplicates from a list:
nums = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(nums))
print(unique)
""",
    },
    # ── 8 ──────────────────────────────────────
    {
        "title": "8. Error Handling",
        "theory": """\
WHY HANDLE ERRORS?
━━━━━━━━━━━━━━━━━━
Runtime errors (exceptions) crash your program.
try/except lets you catch them and respond gracefully.

SYNTAX
━━━━━━
  try:
      risky_code()
  except SomeError as e:
      handle(e)
  except (TypeError, ValueError):
      handle_multiple()
  else:
      # runs only if NO exception occurred
  finally:
      # ALWAYS runs (cleanup, close files, etc.)

COMMON EXCEPTIONS
━━━━━━━━━━━━━━━━━
  ValueError      wrong value type        int("abc")
  TypeError       wrong type operation    "a" + 1
  KeyError        missing dict key        d["nope"]
  IndexError      list out of range       lst[99]
  ZeroDivisionError  division by zero     1/0
  FileNotFoundError  file missing         open("x.txt")
  AttributeError  wrong attribute         None.upper()

RAISING EXCEPTIONS
━━━━━━━━━━━━━━━━━━
  raise ValueError("Age cannot be negative")

CUSTOM EXCEPTIONS
━━━━━━━━━━━━━━━━━
  class InsufficientFundsError(Exception):
      pass

  raise InsufficientFundsError("Balance too low")

ASSERTIONS  (for debugging / invariants)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  assert condition, "message if False"
  assert len(data) > 0, "Data must not be empty"
""",
        "practice": """\
# ── Basic try/except ─────────────────────────
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: cannot divide by zero"
    else:
        return result
    finally:
        print("  (safe_divide called)")

print(safe_divide(10, 2))
print(safe_divide(5, 0))

# ── Multiple exceptions ───────────────────────
def parse_age(text):
    try:
        age = int(text)
        if age < 0 or age > 150:
            raise ValueError("Age out of range")
        return age
    except ValueError as e:
        return f"Invalid: {e}"

print(parse_age("25"))
print(parse_age("abc"))
print(parse_age("-5"))

# ── KeyError & IndexError ─────────────────────
data = {"score": 88}
try:
    print(data["name"])
except KeyError as e:
    print(f"Missing key: {e}")

lst = [1, 2, 3]
try:
    print(lst[10])
except IndexError:
    print("Index out of range")

# ── Custom exception ──────────────────────────
class NegativeBalanceError(Exception):
    pass

def withdraw(balance, amount):
    if amount > balance:
        raise NegativeBalanceError(
            f"Cannot withdraw {amount}, balance is {balance}"
        )
    return balance - amount

try:
    print(withdraw(100, 150))
except NegativeBalanceError as e:
    print(e)
""",
    },
    # ── 9 ──────────────────────────────────────
    {
        "title": "9. Object-Oriented Programming",
        "theory": """\
WHAT IS OOP?
━━━━━━━━━━━━
OOP organises code around objects — bundling data (attributes)
and behaviour (methods) into classes.

FOUR PILLARS
━━━━━━━━━━━━
  1. Encapsulation  – hide internal details
  2. Abstraction    – expose only what's needed
  3. Inheritance    – child class reuses parent code
  4. Polymorphism   – same interface, different behaviour

CLASS SYNTAX
━━━━━━━━━━━━
  class Dog:
      species = "Canis lupus"   # class variable (shared)

      def __init__(self, name, breed):  # constructor
          self.name  = name      # instance variable
          self.breed = breed

      def bark(self):            # instance method
          return f"{self.name} says Woof!"

  rex = Dog("Rex", "Labrador")
  rex.bark()

SPECIAL (DUNDER) METHODS
━━━━━━━━━━━━━━━━━━━━━━━━
  __init__    constructor
  __str__     str(obj)      → human-readable string
  __repr__    repr(obj)     → developer string
  __len__     len(obj)
  __eq__      obj == other
  __lt__      obj < other   (enables sorting)

INHERITANCE
━━━━━━━━━━━
  class GuideDog(Dog):
      def __init__(self, name):
          super().__init__(name, "Labrador")
      def guide(self):
          return f"{self.name} guides the way."

ACCESS MODIFIERS (by convention)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  self.name    public
  self._name   protected  (don't touch from outside)
  self.__name  private    (name-mangled)
""",
        "practice": """\
# ── Basic class ───────────────────────────────
class BankAccount:
    bank_name = "PyBank"   # class variable

    def __init__(self, owner, balance=0):
        self.owner   = owner
        self._balance = balance   # protected

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        return self._balance

    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        return self._balance

    @property
    def balance(self):
        return self._balance

    def __str__(self):
        return f"{self.bank_name} | {self.owner}: ₹{self._balance:,.2f}"

acc = BankAccount("Ravi", 1000)
acc.deposit(500)
acc.withdraw(200)
print(acc)
print(f"Balance: {acc.balance}")

# ── Inheritance ───────────────────────────────
class SavingsAccount(BankAccount):
    def __init__(self, owner, balance=0, rate=0.05):
        super().__init__(owner, balance)
        self.rate = rate

    def add_interest(self):
        interest = self._balance * self.rate
        self._balance += interest
        return interest

savings = SavingsAccount("Meera", 5000, rate=0.06)
earned  = savings.add_interest()
print(f"Interest earned: ₹{earned:.2f}")
print(savings)

# ── Dunder methods ────────────────────────────
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(1, 4)
print(v1 + v2)   # Vector(3, 7)
""",
    },
    # ── 10 ─────────────────────────────────────
    {
        "title": "10. File I/O",
        "theory": """\
READING & WRITING FILES
━━━━━━━━━━━━━━━━━━━━━━━
  open(path, mode) → file object
  Always use with-statement — it closes the file automatically.

MODES
━━━━━
  "r"   read (default)
  "w"   write (overwrites existing)
  "a"   append
  "x"   create new (fails if exists)
  "rb"  read binary    "wb"  write binary

READING
━━━━━━━
  with open("file.txt") as f:
      content = f.read()         # entire file as string
      lines   = f.readlines()    # list of lines
      for line in f:             # iterate line by line (memory-efficient)

WRITING
━━━━━━━
  with open("out.txt", "w") as f:
      f.write("Hello\\n")
      f.writelines(["a\\n", "b\\n"])

CSV FILES
━━━━━━━━━
  import csv
  with open("data.csv") as f:
      reader = csv.DictReader(f)
      for row in reader:
          print(row["name"])

JSON FILES
━━━━━━━━━━
  import json
  with open("data.json") as f:
      data = json.load(f)         # parse JSON → Python dict/list

  with open("out.json","w") as f:
      json.dump(data, f, indent=2)

PATH HANDLING  (use pathlib)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  from pathlib import Path
  p = Path("folder") / "file.txt"
  p.exists()   p.read_text()   p.write_text("hi")
""",
        "practice": """\
import json, io

# ── Writing ───────────────────────────────────
# (using StringIO so we don't write real files here)
buf = io.StringIO()
buf.write("Line 1\\n")
buf.write("Line 2\\n")
buf.write("Line 3\\n")
buf.seek(0)

# ── Reading line by line ──────────────────────
for i, line in enumerate(buf, 1):
    print(f"{i}: {line.rstrip()}")

# ── JSON round-trip ───────────────────────────
data = {
    "students": [
        {"name": "Asha",  "score": 92},
        {"name": "Ravi",  "score": 85},
        {"name": "Priya", "score": 97},
    ]
}

json_str = json.dumps(data, indent=2)
print(json_str)

# Parse back
loaded = json.loads(json_str)
for s in loaded["students"]:
    grade = "A" if s["score"] >= 90 else "B"
    print(f"  {s['name']}: {s['score']} ({grade})")

# ── CSV simulation ────────────────────────────
import csv, io

csv_data = \"\"\"name,age,city
Alice,28,Mumbai
Bob,34,Delhi
Carol,22,Bengaluru\"\"\"

reader = csv.DictReader(io.StringIO(csv_data))
for row in reader:
    print(f"  {row['name']} ({row['age']}) from {row['city']}")
""",
    },
    # ── 11 ─────────────────────────────────────
    {
        "title": "11. Modules & Packages",
        "theory": """\
WHAT IS A MODULE?
━━━━━━━━━━━━━━━━━
A module is simply a .py file. Import it to reuse its code.

IMPORTING
━━━━━━━━━
  import math                    # import whole module
  from math import sqrt, pi      # import specific names
  from math import sqrt as sq    # alias
  import numpy as np             # common alias pattern

STANDARD LIBRARY HIGHLIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  math       sqrt, floor, ceil, log, sin, cos, pi, e
  random     random(), randint(), choice(), shuffle(), sample()
  datetime   date, time, datetime, timedelta
  os         path, getcwd, listdir, makedirs, environ
  sys        argv, path, exit, version
  re         regular expressions: re.match, re.search, re.sub
  collections defaultdict, Counter, deque, OrderedDict
  itertools  product, permutations, combinations, chain
  functools  reduce, lru_cache, partial

CREATING YOUR OWN MODULE
━━━━━━━━━━━━━━━━━━━━━━━━
  # myutils.py
  def greet(name):
      return f"Hi, {name}!"

  # main.py
  from myutils import greet
  print(greet("World"))

PACKAGES
━━━━━━━━
A package is a folder with __init__.py:
  mypackage/
    __init__.py
    module_a.py
    module_b.py

  from mypackage.module_a import something

if __name__ == "__main__"
━━━━━━━━━━━━━━━━━━━━━━━━━
Code under this guard only runs when the file is executed
directly, NOT when imported:
  if __name__ == "__main__":
      main()
""",
        "practice": """\
# ── math ─────────────────────────────────────
import math

print(math.pi)
print(math.sqrt(144))
print(math.floor(3.9), math.ceil(3.1))
print(math.factorial(10))

# ── random ────────────────────────────────────
import random
random.seed(42)   # reproducible

print(random.randint(1, 100))
items = ["red", "blue", "green", "yellow"]
print(random.choice(items))
random.shuffle(items)
print(items)
print(random.sample(range(100), 5))

# ── datetime ──────────────────────────────────
from datetime import datetime, timedelta

now = datetime.now()
print(now.strftime("%d %b %Y  %H:%M"))

birthday = datetime(2000, 6, 15)
today    = datetime.today()
age_days = (today - birthday).days
print(f"Days since 15 Jun 2000: {age_days:,}")

deadline = today + timedelta(days=30)
print(f"Deadline: {deadline.strftime('%d %b %Y')}")

# ── collections.Counter ───────────────────────
from collections import Counter

text   = "python programming is fun and python is great"
counts = Counter(text.split())
print(counts.most_common(3))

# ── itertools ─────────────────────────────────
from itertools import combinations

players = ["A", "B", "C", "D"]
pairs   = list(combinations(players, 2))
print(f"{len(pairs)} pairs:", pairs)
""",
    },
    # ── 12 ─────────────────────────────────────
    {
        "title": "12. Comprehensions & Generators",
        "theory": """\
LIST COMPREHENSION
━━━━━━━━━━━━━━━━━━
  [expression for item in iterable if condition]

  squares = [x**2 for x in range(10)]
  filtered = [x for x in nums if x > 0]
  flat = [n for row in matrix for n in row]

DICT COMPREHENSION
━━━━━━━━━━━━━━━━━━
  {key_expr: val_expr for item in iterable if condition}

  lengths = {w: len(w) for w in words}

SET COMPREHENSION
━━━━━━━━━━━━━━━━━
  {expr for item in iterable}
  unique_lengths = {len(w) for w in words}

GENERATOR EXPRESSIONS
━━━━━━━━━━━━━━━━━━━━━
  (expr for item in iterable)   — uses parentheses

  Generators are lazy (compute on demand) → memory-efficient.
  total = sum(x**2 for x in range(10**7))  # never builds a list

GENERATOR FUNCTIONS
━━━━━━━━━━━━━━━━━━━
  def fibonacci():
      a, b = 0, 1
      while True:
          yield a
          a, b = b, a + b

  gen = fibonacci()
  next(gen)          # 0
  next(gen)          # 1
  [next(gen) for _ in range(8)]   # first 8 fibs

yield
━━━━━
Pauses the function and returns a value.
State is remembered between calls — unlike return.

WHEN TO USE WHICH
━━━━━━━━━━━━━━━━━
  List comp    → small-to-medium data, need it all at once
  Generator    → large/infinite data, process one at a time
""",
        "practice": """\
# ── List comprehensions ───────────────────────
squares = [x**2 for x in range(1, 11)]
print(squares)

matrix = [[1,2,3],[4,5,6],[7,8,9]]
flat   = [n for row in matrix for n in row]
print(flat)

# Pythagorean triples up to 20
triples = [(a,b,c)
           for a in range(1,20)
           for b in range(a,20)
           for c in range(b,20)
           if a**2 + b**2 == c**2]
print(triples)

# ── Dict & set comprehensions ─────────────────
words = ["apple","banana","cherry","date"]
d = {w: w.upper() for w in words if len(w) > 4}
print(d)

s = {len(w) for w in words}
print(s)   # unique lengths

# ── Generator function ────────────────────────
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

gen = fib()
first_10 = [next(gen) for _ in range(10)]
print(first_10)

# ── Memory-efficient generator ────────────────
def evens_up_to(n):
    for i in range(0, n, 2):
        yield i

total = sum(evens_up_to(1001))   # sum of even numbers 0-1000
print(f"Sum of evens 0-1000: {total}")  # 250500

# ── zip & enumerate in comprehensions ─────────
names  = ["Asha", "Ravi", "Meera"]
scores = [88, 92, 79]
report = {n: s for n, s in zip(names, scores)}
print(report)
""",
    },
]


# ─────────────────────────────────────────────
#  UI WIDGETS
# ─────────────────────────────────────────────

def _rounded_bg(widget, r, g, b, a=1, radius=10):
    """Attach a rounded-rectangle background canvas instruction to widget."""
    with widget.canvas.before:
        Color(r, g, b, a)
        rect = RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(radius)])
    widget.bind(
        pos=lambda *_: setattr(rect, "pos", widget.pos),
        size=lambda *_: setattr(rect, "size", widget.size),
    )
    return rect


class TabButton(Button):
    def __init__(self, text, active=False, **kwargs):
        super().__init__(
            text=text,
            font_size=dp(13),
            bold=True,
            background_normal="",
            **kwargs,
        )
        self.set_active(active)

    def set_active(self, active):
        self.background_color = (0.25, 0.45, 0.85, 1) if active else (0.16, 0.18, 0.28, 1)
        self.color = (1, 1, 1, 1)


class CodeLabel(Label):
    """A label tuned for code-like monospace content that wraps correctly."""
    def __init__(self, **kwargs):
        kwargs.setdefault("font_size", dp(12))
        kwargs.setdefault("color", (0.85, 0.92, 1, 1))
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "top")
        kwargs.setdefault("markup", False)
        super().__init__(**kwargs)
        self.bind(width=self._update_text_size, texture_size=self._update_height)

    def _update_text_size(self, *_):
        self.text_size = (self.width, None)

    def _update_height(self, *_):
        self.height = self.texture_size[1]


class LessonDetailScreen(Screen):
    """Full-screen view for one lesson with Theory / Practice tabs."""

    def __init__(self, lesson, **kwargs):
        super().__init__(**kwargs)
        self.lesson = lesson
        self._build()

    def _build(self):
        self.clear_widgets()
        root = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))

        # ── top bar ──────────────────────────────
        top = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))
        back_btn = Button(
            text="← Lessons",
            size_hint=(None, 1),
            width=dp(100),
            background_color=(0.18, 0.20, 0.30, 1),
            background_normal="",
            font_size=dp(12),
        )
        back_btn.bind(on_press=self._go_back)
        top.add_widget(back_btn)

        title_lbl = Label(
            text=self.lesson["title"],
            font_size=dp(14),
            bold=True,
            color=(0.7, 0.85, 1, 1),
            halign="left",
            valign="middle",
        )
        title_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        top.add_widget(title_lbl)
        root.add_widget(top)

        # ── tab row ───────────────────────────────
        tabs = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(6))
        self.theory_tab = TabButton("📖  Theory", active=True, size_hint_x=0.5)
        self.practice_tab = TabButton("⌨️  Practice", active=False, size_hint_x=0.5)
        self.theory_tab.bind(on_press=lambda _: self._show_tab("theory"))
        self.practice_tab.bind(on_press=lambda _: self._show_tab("practice"))
        tabs.add_widget(self.theory_tab)
        tabs.add_widget(self.practice_tab)
        root.add_widget(tabs)

        # ── content area (theory) ─────────────────
        self.theory_scroll = self._make_content_scroll(
            self.lesson["theory"],
            bg=(0.10, 0.12, 0.20, 1),
        )
        self.practice_scroll = self._make_content_scroll(
            self.lesson["practice"],
            bg=(0.07, 0.10, 0.16, 1),
            is_code=True,
        )

        self.content_box = BoxLayout()
        self.content_box.add_widget(self.theory_scroll)
        root.add_widget(self.content_box)
        self.add_widget(root)

    def _make_content_scroll(self, text, bg, is_code=False):
        scroll = ScrollView(do_scroll_x=False, bar_width=dp(4))

        container = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            padding=[dp(12), dp(10)],
            spacing=dp(4),
        )
        container.bind(minimum_height=container.setter("height"))
        _rounded_bg(container, *bg, radius=8)

        lbl = CodeLabel(
            text=text,
            font_size=dp(12) if is_code else dp(13),
        )
        container.add_widget(lbl)
        scroll.add_widget(container)
        return scroll

    def _show_tab(self, which):
        self.content_box.clear_widgets()
        if which == "theory":
            self.theory_tab.set_active(True)
            self.practice_tab.set_active(False)
            self.content_box.add_widget(self.theory_scroll)
        else:
            self.theory_tab.set_active(False)
            self.practice_tab.set_active(True)
            self.content_box.add_widget(self.practice_scroll)

    def _go_back(self, *_):
        App.get_running_app().root.current = "lesson"


# ─────────────────────────────────────────────
#  MAIN LESSON LIST SCREEN
# ─────────────────────────────────────────────

class LessonCard(BoxLayout):
    def __init__(self, lesson, index, on_open, **kwargs):
        super().__init__(orientation="vertical", padding=dp(12), spacing=dp(4), **kwargs)
        self.size_hint_y = None
        self.height = dp(72)
        _rounded_bg(self, 0.13, 0.15, 0.24, radius=10)

        row = BoxLayout(spacing=dp(10))

        # index badge
        badge = Label(
            text=str(index + 1),
            font_size=dp(15),
            bold=True,
            color=(0.4, 0.8, 1, 1),
            size_hint=(None, 1),
            width=dp(32),
        )
        row.add_widget(badge)

        title_lbl = Label(
            text=lesson["title"],
            font_size=dp(14),
            bold=True,
            color=(0.9, 0.95, 1, 1),
            halign="left",
            valign="middle",
        )
        title_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        row.add_widget(title_lbl)

        open_btn = Button(
            text="Open →",
            size_hint=(None, None),
            size=(dp(76), dp(30)),
            background_color=(0.22, 0.40, 0.78, 1),
            background_normal="",
            font_size=dp(12),
            bold=True,
        )
        open_btn.bind(on_press=lambda _: on_open(lesson, index))
        row.add_widget(open_btn)
        self.add_widget(row)


class LessonScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_ui()

    def on_enter(self):
        # Re-build when returning so detail screens are properly named
        pass

    def _build_ui(self):
        self.clear_widgets()
        root = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(10))

        # ── top bar ──────────────────────────────
        top = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back_btn = Button(
            text="← Home",
            size_hint=(None, 1),
            width=dp(90),
            background_color=(0.18, 0.20, 0.30, 1),
            background_normal="",
            font_size=dp(13),
        )
        back_btn.bind(on_press=lambda _: setattr(App.get_running_app().root, "current", "home"))
        top.add_widget(back_btn)

        top.add_widget(Label(
            text="📚  Python Course",
            font_size=dp(20),
            bold=True,
            color=(1, 1, 1, 1),
            halign="left",
        ))
        root.add_widget(top)

        # sub-header
        root.add_widget(Label(
            text=f"{len(LESSONS)} lessons  •  tap any lesson to open theory & practice",
            font_size=dp(12),
            color=(0.55, 0.60, 0.80, 1),
            halign="left",
            size_hint_y=None,
            height=dp(20),
        ))

        # ── lesson list ──────────────────────────
        scroll = ScrollView(do_scroll_x=False, bar_width=dp(4))
        grid = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
            size_hint_y=None,
            padding=[0, 0, 0, dp(20)],
        )
        grid.bind(minimum_height=grid.setter("height"))

        for i, lesson in enumerate(LESSONS):
            card = LessonCard(lesson, i, on_open=self._open_lesson)
            grid.add_widget(card)

        scroll.add_widget(grid)
        root.add_widget(scroll)
        self.add_widget(root)

    def _open_lesson(self, lesson, index):
        sm = App.get_running_app().root
        screen_name = f"lesson_detail_{index}"

        if not sm.has_screen(screen_name):
            detail = LessonDetailScreen(lesson, name=screen_name)
            sm.add_widget(detail)

        sm.current = screen_name
