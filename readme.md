# Functional Programming Jargon

Functional programming (FP) provides many advantages, and its popularity has been increasing as a result. However, each programming paradigm comes with its own unique jargon and FP is no exception. By providing a glossary, we hope to make learning FP easier.

This is a fork of [Functional Programming Jargon](https://github.com/jmesyou/functional-programming-jargon). Examples are presented in Python3.

This document attempts to adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/) as best as possible.

This document is WIP and pull requests are welcome!

__Translations__
* [Portuguese](https://github.com/alexmoreno/jargoes-programacao-funcional)
* [Spanish](https://github.com/idcmardelplata/functional-programming-jargon/tree/master)
* [Chinese](https://github.com/shfshanyue/fp-jargon-zh)
* [Bahasa Indonesia](https://github.com/wisn/jargon-pemrograman-fungsional)
* [Scala World](https://github.com/ikhoon/functional-programming-jargon.scala)
* [Korean](https://github.com/sphilee/functional-programming-jargon)

__Table of Contents__
<!-- RM(noparent,notop) -->

* [Arity](#arity)
* [Higher-Order Functions (HOF)](#higher-order-functions-hof)
* [Closure](#closure)
* [Partial Application](#partial-application)
* [Currying](#currying)
* [Auto Currying](#auto-currying)
* [Function Composition](#function-composition)
* [Continuation](#continuation)
* [Purity](#purity)
* [Side effects](#side-effects)
* [Idempotent](#idempotent)
* [Point-Free Style](#point-free-style)
* [Predicate](#predicate)
* [Contracts](#contracts)
* [Category](#category)
* [Value](#value)
* [Constant](#constant)
* [Functor](#functor)
* [Pointed Functor](#pointed-functor)
* [Lift](#lift)
* [Referential Transparency](#referential-transparency)
* [Equational Reasoning](#equational-reasoning)
* [Lambda](#lambda)
* [Lambda Calculus](#lambda-calculus)
* [Lazy evaluation](#lazy-evaluation)
* [Monoid](#monoid)
* [Monad](#monad)
* [Comonad](#comonad)
* [Applicative Functor](#applicative-functor)
* [Morphism](#morphism)
  * [Endomorphism](#endomorphism)
  * [Isomorphism](#isomorphism)
  * [Homomorphism](#homomorphism)
  * [Catamorphism](#catamorphism)
  * [Anamorphism](#anamorphism)
  * [Hylomorphism](#hylomorphism)
  * [Paramorphism](#paramorphism)
  * [Apomorphism](#apomorphism)
* [Setoid](#setoid)
* [Semigroup](#semigroup)
* [Foldable](#foldable)
* [Lens](#lens)
* [Type Signatures](#type-signatures)
* [Algebraic data type](#algebraic-data-type)
  * [Sum type](#sum-type)
  * [Product type](#product-type)
* [Option](#option)
* [Function](#function)
* [Partial function](#partial-function)
* [Functional Programming Libraries in Python](#functional-programming-libraries-in-javascript)


<!-- /RM -->

## Arity

The number of arguments a function takes. From words like unary, binary, ternary, etc. This word has the distinction of being composed of two suffixes, "-ary" and "-ity." Addition, for example, takes two arguments, and so it is defined as a binary function or a function with an arity of two. Such a function may sometimes be called "dyadic" by people who prefer Greek roots to Latin. Likewise, a function that takes a variable number of arguments is called "variadic," whereas a binary function must be given two and only two arguments, currying and partial application notwithstanding (see below).

```python
from inspect import signature 

add = lambda a, b: a + b

arity = len(signature(add).parameters)
print(arity) # 2

# The arity of add is 2
```

## Higher-Order Functions (HOF)

A function which takes a function as an argument and/or returns a function.

```python
filter = lambda predicate, xs: [x for x in xs if predicate(xs)] 
```

```python
is_a = lambda T: lambda x: type(x) is T
```

```python
filter(is_a(int), [0, '1', 2, None]) # [0, 2]
```

## Closure

A closure is a way of accessing a variable outside its scope.
Formally, a closure is a technique for implementing lexically scoped named binding. It is a way of storing a function with an environment.

A closure is a scope which captures local variables of a function for access even after the execution has moved out of the block in which it is defined.
ie. they allow referencing a scope after the block in which the variables were declared has finished executing.


```python
add_to = lambda x: lambda y: x + y
add_to_five = add_to(5)
add_to_five(3) # returns 8
```
The function ```add_to()``` returns a function(internally called ```add()```), lets store it in a variable called ```add_to_five``` with a curried call having parameter 5.

Ideally, when the function ```add_to``` finishes execution, its scope, with local variables add, x, y should not be accessible. But, it returns 8 on calling ```add_to_five()```. This means that the state of the function ```add_to``` is saved even after the block of code has finished executing, otherwise there is no way of knowing that ```add_to``` was called as ```add_to(5)``` and the value of x was set to 5.

Lexical scoping is the reason why it is able to find the values of x and add - the private variables of the parent which has finished executing. This value is called a Closure.

The stack along with the lexical scope of the function is stored in form of reference to the parent. This prevents the closure and the underlying variables from being garbage collected(since there is at least one live reference to it).

Lambda Vs Closure: A lambda is essentially a function that is defined inline rather than the standard method of declaring functions. Lambdas can frequently be passed around as objects.

A closure is a function that encloses its surrounding state by referencing fields external to its body. The enclosed state remains across invocations of the closure.

__Further reading/Sources__
* [Lambda Vs Closure](http://stackoverflow.com/questions/220658/what-is-the-difference-between-a-closure-and-a-lambda)
* [JavaScript Closures highly voted discussion](http://stackoverflow.com/questions/111102/how-do-javascript-closures-work)

## Partial Application

Partially applying a function means creating a new function by pre-filling some of the arguments to the original function.


```python
# Helper to create partially applied functions
# Takes a function and some arguments
partial = lambda f, *args: 
  # returns a function that takes the rest of the arguments
  lambda *more_args:
    # and calls the original function with all of them
    f(args, more_args)
# Something to apply
add3 = lambda a, b, c: a + b + c
# Partially applying `2` and `3` to `add3` gives you a one-argument function
five_plus = partial(add3, 2, 3) # (c) => 2 + 3 + c

five_plus(4) # 9
```

You can also use `functools.partial` to partially apply a function in Python:

```python
from functools import partial 

add_more = partial(add3, 2, 3) # (c) => 2 + 3 + c
```

Partial application helps create simpler functions from more complex ones by baking in data when you have it. [Curried](#currying) functions are automatically partially applied.


## Currying

The process of converting a function that takes multiple arguments into a function that takes them one at a time.

Each time the function is called it only accepts one argument and returns a function that takes one argument until all arguments are passed.

```python
sum = lambda a, b: a + b 

curried_sum = lambda a: lambda b: a + b

curried_sum(40)(2) # 42.

add2 = curried_sum(2) # (b) => 2 + b

add2(10) # 12
```

## Auto Currying
Transforming a function that takes multiple arguments into one that if given less than its correct number of arguments returns a function that takes the rest. When the function gets the correct number of arguments it is then evaluated.

The `toolz` module has an currying decorator which works this way

```python
from toolz import curry 

@curry
def add(x, y): return x + y 

add(1, 2) # 3
add(1)    # (y) => 1 + y
add(1)(2) # 3
```

__Further reading__
* [Favoring Curry](http://fr.umio.us/favoring-curry/)
* [Hey Underscore, You're Doing It Wrong!](https://www.youtube.com/watch?v=m3svKOdZijA)

## Function Composition

The act of putting two functions together to form a third function where the output of one function is the input of the other.

```python
import math 

compose = lambda f, g: lambda a: f(g(a)) # Definition
floor_and_str = compose(str, Math.floor) # Usage
floor_and_string(121.212121) # '121'
```

## Continuation

At any given point in a program, the part of the code that's yet to be executed is known as a continuation.

```python
print_as_string = lambda num: print(num)

# this was previously defined as multi-line function in Javascript
# however, Python only supports single expression lambdas
add_one_and_continue = lambda num, cc: cc(num + 1) 

add_one_and_continue(2, print_as_string) # 'Given 3'
```

Continuations are often seen in asynchronous programming when the program needs to wait to receive data before it can continue. The response is often passed off to the rest of the program, which is the continuation, once it's been received.

```python
continue_program_with = lambda data: ... # Continues program with data

read_file_async('path/to/file', lambda err, response: raise err if err else continue_program_with(response))
```

## Purity

A function is pure if the return value is only determined by its
input values, and does not produce side effects.

```python
greet = lambda name: 'Hi, {}'.format(name)

greet('Brianne') # 'Hi, Brianne'
```

As opposed to each of the following:

```python
name = 'Brianne'

greet = lambda: 'Hi, {}'.format(name)

greet() # "Hi, Brianne"
```

The above example's output is based on data stored outside of the function...

```python
greeting = None

def greet(name):
  global greeting
  greeting = 'Hi, {}'.format(name)

greet('Brianne')
greeting # "Hi, Brianne"
```

... and this one modifies state outside of the function.

## Side effects

A function or expression is said to have a side effect if apart from returning a value, it interacts with (reads from or writes to) external mutable state.

```python
different_ever_time = list()
```

```python
print('IO is a side effect!')
```

## Idempotent

A function is idempotent if reapplying it to its result does not produce a different result.

```
f(f(x)) ≍ f(x)
```

```python
import math

math.abs(math.abs(10))
```

```python
sorted(sorted(sorted([2, 1])))
```

## Point-Free Style

Writing functions where the definition does not explicitly identify the arguments used. This style usually requires [currying](#currying) or other [Higher-Order functions](#higher-order-functions-hof). A.K.A Tacit programming.

```python
# Given
map = lambda fn: lambda xs: [fn(x) for x in xs]
add = lambda a: lambda b: a + b

# Then

# Not points-free - `numbers` is an explicit argument
increment_all = lambda numbers: map(add(1))(numbers)

# Points-free - The list is an implicit argument
increment_all2 = map(add(1))
```

`increment_all` identifies and uses the parameter `numbers`, so it is not points-free.  `increment_all2` is written just by combining functions and values, making no mention of its arguments.  It __is__ points-free.

Points-free function definitions look just like normal assignments without `def` or `lambda`.

## Predicate
A predicate is a function that returns true or false for a given value. A common use of a predicate is as the callback for array filter.

```python
predicate = lambda a: a > 2

filter(predicate, [1, 2, 3, 4]) # [3, 4]
```

## Contracts

A contract specifies the obligations and guarantees of the behavior from a function or expression at runtime. This acts as a set of rules that are expected from the input and output of a function or expression, and errors are generally reported whenever a contract is violated.

```python
def throw(ex):
  raise ex

# Define our contract : int -> boolean
contract = lambda value: True if type(value) is int else throw(Exception('Contract violated: expected int -> boolean'))

add1 = lambda num: contract(num) and num + 1

add1(2) # 3
add1('some string') # Contract violated: expected int -> boolean
```

## Category

A category in category theory is a collection of objects and morphisms between them. In programming, typically types
act as the objects and functions as morphisms.

To be a valid category 3 rules must be met:

1. There must be an identity morphism that maps an object to itself.
    Where `a` is an object in some category,
    there must be a function from `a -> a`.
2. Morphisms must compose.
    Where `a`, `b`, and `c` are objects in some category,
    and `f` is a morphism from `a -> b`, and `g` is a morphism from `b -> c`;
    `g(f(x))` must be equivalent to `(g • f)(x)`.
3. Composition must be associative
    `f • (g • h)` is the same as `(f • g) • h`

Since these rules govern composition at very abstract level, category theory is great at uncovering new ways of composing things.

__Further reading__

* [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/)

## Value

Anything that can be assigned to a variable.

```python
from collections import namedtuple
Person = namedtuple('Person', 'name age')
5
Person('John', 30)
lambda a: a
[1]
None
```

## Constant

A variable that cannot be reassigned once defined.

```python

five = 5
john = Person('John', 30)
```

Constants are [referentially transparent](#referential-transparency). That is, they can be replaced with the values that they represent without affecting the result.

With the above two constants the following expression will always return `true`.

```python
john.age + five == Person('John' 30).age + 5
```

## Functor

An object that implements a `map` function which, while running over each value in the object to produce a new object, adheres to two rules:

__Preserves identity__
```
object.map(x => x) ≍ object
```

__Composable__

```
object.map(compose(f, g)) ≍ object.map(g).map(f)
```

(`f`, `g` are arbitrary functions)

Iterables in Python are functors since they abide by the two functor rules:

```python

map(lambda x: x, [1, 2, 3]) # = [1, 2, 3]
map(lambda x: x, (1, 2, 3)) # = (1, 2, 3)
```

and

```python
f = lambda x: x + 1
g = lambda x: x * 2

map(lambda x: f(g(x)), [1, 2, 3]) # = [3, 5, 7]
map(f, map(g, [1, 2, 3]))         # = [3, 5, 7]
```

## Pointed Functor
An object with an `of` function that puts _any_ number of values into it. The following property
`of(f(x)) == of(x).map(f)` must also hold for any pointed functor.

We create a custom class `Array` which mimics the original Javascript for a pointed functor.

```python 

class Array(list):

  of = lambda *args: Array([a for a in args])

Array.of(1) # [1]
```

## Lift

Lifting is when you take a value and put it into an object like a [functor](#pointed-functor). If you lift a function into an [Applicative Functor](#applicative-functor) then you can make it work on values that are also in that functor.

Some implementations have a function called `lift`, or `liftA2` to make it easier to run functions on functors.

```python
# OSlash applicative library https://github.com/dbrattli/oslash
from oslash import List, Applicative # we import the Applicative 

liftA2 = lambda f: lambda a, b:  a.map(f) * b # a, b is of type Applicative where (*) is the apply function.

mult = lambda a: lambda b: a * b # (*) is just regular multiplication

lifted_mult = liftA2(mult) # this function now works on functors like oslash.List

lifted_mult(List([1, 2]), List([3])) # [3, 6]
liftA2(lambda a: lambda b: a + b)(List([1, 2]), List([3, 4])) # [4, 5, 5, 6]

# or using oslash, it can alternatively be written as

List([1, 2]).lift_a2(mult, List([3]))
List([1, 2]).lift_a2(lambda a: lambda b: a + b, List([3, 4]))
```

Lifting a one-argument function and applying it does the same thing as `map`.

```python
increment = lambda x: x + 1

lift(increment)(List([2])) # [3]
List([2]).map(increment)  # [3]
```


## Referential Transparency

An expression that can be replaced with its value without changing the
behavior of the program is said to be referentially transparent.

Say we have function greet:

```python
greet = lambda: 'Hello World!'
```

Any invocation of `greet()` can be replaced with `Hello World!` hence greet is
referentially transparent.

##  Equational Reasoning

When an application is composed of expressions and devoid of side effects, truths about the system can be derived from the parts.

## Lambda

An anonymous function that can be treated like a value.

```python 
def f(a):
  return a + 1

lambda a: a + 1
```
Lambdas are often passed as arguments to Higher-Order functions.

```python
List([1, 2]).map(lambda x: x + 1) # [2, 3]
```

You can assign a lambda to a variable.

```python
add1 = lambda a: a + 1
```

## Lambda Calculus
A branch of mathematics that uses functions to create a [universal model of computation](https://en.wikipedia.org/wiki/Lambda_calculus).

## Lazy evaluation

Lazy evaluation is a call-by-need evaluation mechanism that delays the evaluation of an expression until its value is needed. In functional languages, this allows for structures like infinite lists, which would not normally be available in an imperative language where the sequencing of commands is significant.

```python
import random
def rand(): 
  while True:
    yield random.randint(1,101)
```

```python
randIter = rand()
next(randIter) # Each execution gives a random value, expression is evaluated on need.
```

## Monoid

An object with a function that "combines" that object with another of the same type.

One simple monoid is the addition of numbers:

```python
1 + 1 # 2
```
In this case number is the object and `+` is the function.

An "identity" value must also exist that when combined with a value doesn't change it.

The identity value for addition is `0`.
```python
1 + 0 # 1
```

It's also required that the grouping of operations will not affect the result (associativity):

```python
1 + (2 + 3) == (1 + 2) + 3 # true
```

List concatenation also forms a monoid:

```python
[1, 2] + [3, 4] # [1, 2, 3, 4]
```

The identity value is empty array `[]`

```python
[1, 2] + [] # [1, 2]
```

If identity and compose functions are provided, functions themselves form a monoid:

```python
identity = lambda a: a
compose = lambda f, g: lambda x: f(g(x))
```
`foo` is any function that takes one argument.
```
compose(foo, identity) ≍ compose(identity, foo) ≍ foo
```

## Monad

A monad is an object with [`of`](#pointed-functor) and `chain` functions. `chain` is like [`map`](#functor) except it un-nests the resulting nested object.

```python
# pymonad Monad library https://bitbucket.org/jason_delaat/pymonad
from pymonad.List import *
from functools import reduce
#implementation
class Array(list):

  def of(*args): 
    return Array([a for a in args])

  def chain(self, f):
    return reduce(lambda acc, it: acc + f(it), self[:], [])

  def map(self, f):
    return [f(x) for x in self[:]]

# Usage
Array.of('cat,dog', 'fish,bird').chain(lambda a: a.split(',')) # ['cat', 'dog', 'fish', 'bird']

# Contrast to map
Array.of('cat,dog', 'fish,bird').map(lambda a: a.split(',')) # [['cat', 'dog'], ['fish', 'bird']]
```

`of` is also known as `return` in other functional languages.
`chain` is also known as `flatmap` and `bind` in other languages.

## Comonad

An object that has `extract` and `extend` functions.

```python
class CoIdentity:

  def __init__(self, v):
    self.val = v

  def extract(self):
    return self.val

  def extend(f):
    return CoIdentity(f(self))
```

Extract takes a value out of a functor.

```python
CoIdentity(1).extract() # 1
```

Extend runs a function on the comonad. The function should return the same type as the comonad.

```python
CoIdentity(1).extend(lambda co: co.extract() + 1) # CoIdentity(2)
```

## Applicative Functor

An applicative functor is an object with an `ap` function. `ap` applies a function in the object to a value in another object of the same type.

```python
from functools import reduce
# Implementation

class Array(list):

  def of(*args): 
    return Array([a for a in args])

  def chain(self, f):
    return reduce(lambda acc, it: acc + f(it), self[:], [])

  def map(self, f):
    return [f(x) for x in self[:]]

  def ap(self, xs):
    return reduce(lambda acc, f: acc + (xs.map(f)), self[:], [])

# Example usage
Array([lambda a: a + 1]).ap(Array([1])) # [2]
```

This is useful if you have two objects and you want to apply a binary function to their contents.

```python
# Arrays that you want to combine
arg1 = [1, 3]
arg2 = [4, 5]

# combining function - must be curried for this to work
add = lambda x: lambda y: x + y

partially_applied_adds = [add].ap(arg1) # [(y) => 1 + y, (y) => 3 + y]
```

This gives you an array of functions that you can call `ap` on to get the result:

```python
partially_applied_adds.ap(arg2) # [5, 6, 7, 8]
```

## Morphism

A transformation function.

### Endomorphism

A function where the input type is the same as the output.

```python
# uppercase :: String -> String
uppercase = lambda s: s.upper() 

# decrement :: Number -> Number
decrement = lambda x: x - 1
```

### Isomorphism

A pair of transformations between 2 types of objects that is structural in nature and no data is lost.

For example, 2D coordinates could be stored as an array `[2,3]` or object `{x: 2, y: 3}`.

```python
# Providing functions to convert in both directions makes them isomorphic.
from collections import namedtuple

Coords = namedtuple('Coords', 'x y')
pair_to_coords = lambda pair: Coords(pair[0], pair[1])

coords_to_pair = lambda coords: [coords.x, coords.y]

coords_to_pair(pair_to_coords([1, 2])) # [1, 2]

pair_to_coords(coords_to_pair(Coords(1, 2))) # Coords(x=1, y=2)
```

### Homomorphism

A homomorphism is just a structure preserving map. In fact, a functor is just a homomorphism between categories as it preserves the original category's structure under the mapping.

```python
from pymonad import *
f * A.unit(x) == A.unit(f(x))

(lambda x: x.upper()) * (Either.unit("oreos")) == Either.unit("oreos".upper())
```

### Catamorphism

A `reduce_right` function that applies a function against an accumulator and each value of the array (from right-to-left) to reduce it to a single value.

```python
from functools import reduce
class Array(list):

  def reduce_right(self, f, init):
    return reduce(f, self[::-1], init)

sum = lambda xs: xs.reduce_right(lambda acc, x: acc + x, 0)

sum(Array([1, 2, 3, 4, 5])) # 15
```

### Anamorphism

An `unfold` function. An `unfold` is the opposite of `fold` (`reduce`). It generates a list from a single value.

```python

def unfold(f, seed):
  def go(f, seed, acc):
    res = f(seed)
    return go(f, res[1], acc + [res[0]]) if res else acc

  return go(f, seed, [])
```

```python
count_down = lambda n: unfold(lambda n: None if n <= 0 else (n, n - 1), n)

count_down(5) # [5, 4, 3, 2, 1]
```

### Hylomorphism

The combination of anamorphism and catamorphism.

### Paramorphism

A function just like `reduce_right`. However, there's a difference:

In paramorphism, your reducer's arguments are the current value, the reduction of all previous values, and the list of values that formed that reduction.

```python

def para(reducer, accumulator, elements):
  if not len(elements):
    return accumulator

  head = elements[0]
  tail = elements[1:]

  return reducer(head, tail, para(reducer, accumulator, tail))

suffixes = lambda lst: para(lambda x, xs, suffxs: [xs, *suffxs], [], lst)

suffixes([1, 2, 3, 4, 5]) # [[2, 3, 4, 5], [3, 4, 5], [4, 5], [5], []]
```

The third parameter in the reducer (in the above example, `[x, *xs]`) is kind of like having a history of what got you to your current acc value.

### Apomorphism

it's the opposite of paramorphism, just as anamorphism is the opposite of catamorphism. Whereas with paramorphism, you combine with access to the accumulator and what has been accumulated, apomorphism lets you `unfold` with the potential to return early.

## Setoid

An object that has an `equals` function which can be used to compare other objects of the same type.

Make array a setoid:

```python 

class Array(list):

  def equals(self, other):
    if len(self) != len(other):
      return False 
    else:
      return reduce(lambda ident, pair: ident and (pair[0] == pair[1]), zip(self[:], other[:]), True)

Array([1, 2]).equals(Array([1, 2])) # true
Array([1, 2]).equals(Array([0])) # false
```

## Semigroup

An object that has a `concat` function that combines it with another object of the same type.

```python
[1] + [2] # [1, 2]
```

## Foldable

An object that has a `reduce` function that applies a function against an accumulator and each element in the array (from left to right) to reduce it to a single value.

```python
sum = lambda lst: reduce(lambda acc, x: acc + x, lst, 0)
sum([1, 2, 3]) # 6
```

## Lens ##
A lens is a structure (often an object or function) that pairs a getter and a non-mutating setter for some other data
structure.

```python
from lenses import lens
# Using [python-lenses](https://python-lenses.readthedocs.io/en/latest/tutorial/methods.html)

name_lens = lens['name'] # we create an unbound lens which accesses the value associate with the 'name' key of a dict
```

Having the pair of get and set for a given data structure enables a few key features.

```python
person = {'name': 'Gertrude Blanch'}

# invoke the getter
name_lens.get()(person) # 'Gertrude Blanch'

# invoke the setter
name_lens.set('Shafi Goldwasser')(person) # {'name': 'Shafi Goldwasser'}

# run a function on the value in the structure
name_lens.modify(lambda x: x.upper())(person) # {'name': 'GERTRUDE BLANCH'}
```

Lenses are also composable. This allows easy immutable updates to deeply nested data.

```python
# This lens focuses on the first item in a non-empty array
first_lens = lens[0]

people = [{'name': 'Gertrude Blanch'}, {'name': 'Shafi Goldwasser'}]

# Despite what you may assume, lenses compose left-to-right.
(first_lens & name_lens).modify(lambda x: x.upper())(people) # [{'name': 'GERTRUDE BLANCH'}, {'name': 'Shafi Goldwasser'}]
```

## Type Signatures

Often functions in JavaScript will include comments that indicate the types of their arguments and return values.

There's quite a bit of variance across the community but they often follow the following patterns:

```python
# function :: a -> b -> c

# add :: int -> int -> int
# alternatively could be float -> float -> float
add = lambda y: lambda x: x + y

# increment :: int -> int
increment = lambda x: x + 1
```

If a function accepts another function as an argument it is wrapped in parentheses.

```python
# call :: (a -> b) -> a -> b
call = lambda f: lambda x: f(x)
```

The letters `a`, `b`, `c`, `d` are used to signify that the argument can be of any type. The following version of `map` takes a function that transforms a value of some type `a` into another type `b`, an array of values of type `a`, and returns an array of values of type `b`.

```python
# map :: (a -> b) -> [a] -> [b]
map(f, lst)
```

__Further reading__
* [Ramda's type signatures](https://github.com/ramda/ramda/wiki/Type-Signatures)
* [Mostly Adequate Guide](https://drboolean.gitbooks.io/mostly-adequate-guide/content/ch7.html#whats-your-type)
* [What is Hindley-Milner?](http://stackoverflow.com/a/399392/22425) on Stack Overflow

## Algebraic data type
A composite type made from putting other types together. Two common classes of algebraic types are [sum](#sum-type) and [product](#product-type).

### Sum type
A Sum type is the combination of two types together into another one. It is called sum because the number of possible values in the result type is the sum of the input types.

JavaScript doesn't have types like this (neither does python) but we can use `Set`s to pretend:
```python
# imagine that rather than sets here we have types that can only have these values
bools = set([True, False])
half_true = set(['half-true'])

# The weakLogic type contains the sum of the values from bools and halfTrue
weak_logic_values = bools.union(half_true)
```

Sum types are sometimes called union types, discriminated unions, or tagged unions.

The [sumtypes](https://github.com/radix/sumtypes/) library in Python helps with defining and using union types.

### Product type

A **product** type combines types together in a way you're probably more familiar with:

```python
from collections import namedtuple

Point = namedtuple('Point', 'x y')
# point :: (Number, Number) -> {x: Number, y: Number}
point = lambda pair: Point(x, y)
```
It's called a product because the total possible values of the data structure is the product of the different values. Many languages have a tuple type which is the simplest formulation of a product type.

See also [Set theory](https://en.wikipedia.org/wiki/Set_theory).

## Option
Option is a [sum type](#sum-type) with two cases often called `Some` and `None`.

Option is useful for composing functions that might not return a value.

```python
# Naive definition

class _Some:
  
  def __init__(self, v):
    self.val = v

  def map(self, f):
    return _Some(f(self.val))

  def chain(self, f):
    return f(self.val)
  
# None is a keyword in python
class _None:

  def map(self, f):
    return self

  def chain(self, f):
    return self

# maybe_prop :: String -> (String => a) -> Option a
maybe_prop = lambda key, obj: _None() if key not in obj else Some(obj[key])
```
Use `chain` to sequence functions that return `Option`s
```python

# get_item :: Cart -> Option CartItem
get_item = lambda cart: maybe_prop('item', cart)

# get_price :: Cart -> Option Float
get_price = lambda item: maybe_prop('price', item)

# get_nested_price :: Cart -> Option a
get_nested_price = lambda cart: get_item(cart).chain(get_price)

get_nested_price({}) # _None()
get_nested_price({"item": {"foo": 1}}) # _None()
get_nested_price({"item": {"price": 9.99}}) # _Some(9.99)
```

`Option` is also known as `Maybe`. `Some` is sometimes called `Just`. `None` is sometimes called `Nothing`.

## Function
A **function** `f :: A => B` is an expression - often called arrow or lambda expression - with **exactly one (immutable)** parameter of type `A` and **exactly one** return value of type `B`. That value depends entirely on the argument, making functions context-independant, or [referentially transparent](#referential-transparency). What is implied here is that a function must not produce any hidden [side effects](#side-effects) - a function is always [pure](#purity), by definition. These properties make functions pleasant to work with: they are entirely deterministic and therefore predictable. Functions enable working with code as data, abstracting over behaviour:

```python
# times2 :: Number -> Number
times2 = lambda n: n * 2

map(times2, [1, 2, 3] # [2, 4, 6]
```

## Partial function
A partial function is a [function](#function) which is not defined for all arguments - it might return an unexpected result or may never terminate. Partial functions add cognitive overhead, they are harder to reason about and can lead to runtime errors. Some examples:
```python
from functools import partial
# example 1: sum of the list
# sum :: [int] -> int
sum = partial(reduce, lambda a, b: a + b, arr)
sum([1, 2, 3]) # 6
sum([])        # TypeError: reduce() of empty sequence with no initial value

# example 2: get the first item in list
# first :: [a] -> a
first = lambda a: a[0]
first([42]) # 42
first([])   # IndexError
# or even worse:
first([[42]])[0] # 42
first([])[0]     # Uncaught TypeError: an IndexError is throw instead

# example 3: repeat function N times
# times :: int -> (int -> int) -> int
times = lambda n: lambda fn: n and (fn(n), times(n - 1)(fn))
times(3)(print)
# 3
# 2
# 1
# out: (None, (None, (None, 0)))
times(-1)(print)
# RecursionError: maximum recursion depth exceeded while calling a Python object
```

### Dealing with partial functions
Partial functions are dangerous as they need to be treated with great caution. You might get an unexpected (wrong) result or run into runtime errors. Sometimes a partial function might not return at all. Being aware of and treating all these edge cases accordingly can become very tedious.
Fortunately a partial function can be converted to a regular (or total) one. We can provide default values or use guards to deal with inputs for which the (previously) partial function is undefined. Utilizing the [`Option`](#Option) type, we can yield either `Some(value)` or `None` where we would otherwise have behaved unexpectedly:
```python

# re-order function arguments for easier partial function application
_reduce = lambda fn, init, seq: reduce(fn, seq, init) 
# example 1: sum of the list
# we can provide default value so it will always return result
# sum :: [int] -> int

sum = partial(_reduce, lambda a, b: a + b, 0)
sum([1, 2, 3]) # 6
sum([]) # 0

# example 2: get the first item in list
# change result to Option
# first :: [A] -> Option A
first = lambda a: _Some(a[0]) if len(a) else _None()
first([42]).map(print) # 42
first([]).map(print) # print won't execute at all
# our previous worst case
first([[42]]).map(lambda a: print(a[0])) # 42
first([]).map(lambda a: print(a[0])) # won't execute, so we won't have error here
# more of that, you will know by function return type (Option)
# that you should use `.map` method to access the data and you will never forget
# to check your input because such check become built-in into the function

# example 3: repeat function N times
# we should make function always terminate by changing conditions:
# times :: int -> (int -> int) -> int
times = lambda n: lambda fn: n > 0 and (fn(n), times(n - 1)(fn))
times(3)(print)
# 3
# 2
# 1
times(-1)(print)
# won't execute anything
```
Making your partial functions total ones, these kinds of runtime errors can be prevented. Always returning a value will also make for code that is both easier to maintain as well as to reason about.

## Functional Programming Libraries in Python

### In This Doc
  * [functools](https://docs.python.org/3/library/functools.html)
  * [oslash](https://github.com/dbrattli/oslash)
  * [python-lenses](https://github.com/ingolemo/python-lenses)
  * [pymonad](https://bitbucket.org/jason_delaat/pymonad)
  * [toolz](https://github.com/pytoolz/toolz)

A comprehensive curated list of functional programming libraries for Python can be found [here](https://github.com/sfermigier/awesome-functional-python)

---

__P.S:__ This repo is successful due to the wonderful [contributions](https://github.com/hemanth/functional-programming-jargon/graphs/contributors)!
