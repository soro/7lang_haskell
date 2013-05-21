\documentclass[9pt]{beamer}

\usepackage{minted}
\usepackage{hyperref}
\usepackage{upquote}

\title{
Haskell\\
or\\
functional purity and laziness 
}
\author{Soeren Roerden}
\date

%\setbeamertemplate{itemize item}{}
\setbeamersize{text margin left=6mm}

\begin{document}

\AtBeginSection[]
{
  \begin{frame}
    \begin{center}
        \Large{\insertsection}
    \end{center}
  \end{frame}
}

\frame{\titlepage}

% --------------------

\begin{frame}[fragile]{Preliminaries}
\begin{code}
{-# LANGUAGE BangPatterns #-}
module Intro where
import Test.QuickCheck
import Test.HUnit
import Text.Show.Functions
import Control.DeepSeq
\end{code}

\end{frame}

\section{Purity}

% -------------------

\begin{frame}{Purity}

\begin{center}
Pure computations always result in the same value given the same inputs,
meaning they can not perform side effects such as mutation of assignables or
I/O.
\end{center}

\end{frame}

% --------------------

\begin{frame}[fragile]{Purity}
\begin{minted}{javascript}
var impure = function(x, y) {
  console.log("foobar");
  return x + y;
}
\end{minted}
\begin{minted}{javascript}
var pure = function(x, y) {
  return x + y;
}
\end{minted}
\end{frame}


% --------------------

\section{Syntax and Type Signatures}

% --------------------

\begin{frame}[fragile]{Syntax and Type Signatures}
\begin{itemize}

\item Haskell has white space sensitive syntax, just like Python.

\item<2-> Type for cons - prepending an element to a list
\begin{minted}{haskell}
Prelude> :t (:)
(:) :: a -> [a] -> [a]
\end{minted}

\item<3-> Haskell functions are curried by default
\begin{minted}{haskell}
Prelude> :t (+)
(+) :: Num a => a -> a -> a

Prelude> :t uncurry (+)
uncurry (+) :: Num c => (c, c) -> c
\end{minted}

\item<4-> Juxtaposition is function application and has highest fixity. Use $\$$ for least fixity.
\begin{minted}{haskell}
Prelude> (*) 5 $ 3 + 2
25
Prelude> (*) 5 3 + 2
17
\end{minted}
  
\end{itemize}
\end{frame}


% --------------------

\begin{frame}[fragile]{Syntax and Type Signatures}
\begin{itemize}

\item Anonymous functions can be defined via the syntax
\begin{minted}{haskell}
\x -> x + 1
\end{minted}
 
\item<2-> Function definition
\begin{code}
add1 x = x + 1
add1' = (+ 1) 
add1'' = \x -> x + 1
\end{code}

\item<3-> Ranges and list comprehensions
\begin{code}
twoToFour = [2..4]
twoToFour' = [x + 1 | x <- [1..3]]
\end{code}

\item<4-> HUnit tests
\begin{code}
map_test = map add1 [1,2,3] ~=? twoToFour
map_test' = map add1' [1,2,3] ~=? twoToFour'
\end{code}

\end{itemize}
\end{frame}

% --------------------

\begin{frame}[fragile]{Syntax and Type Signatures}

\begin{itemize}
\item \`{}\textit{fun}\`{} changes fixity of \textit{fun} from prefix to infix, (\textit{fun}) does the opposite
\begin{code}
concat' a b = a ++ b
ctest = [concat' "foo" "bar" ~=? "foobar",
         "foo" `concat'` "bar" ~=? "foobar"]
\end{code}

\item<2-> Pattern matching allows one to define functions by (exhaustive) cases
\begin{code}
doubleVision :: [a] -> [a]
doubleVision (x:xs) = x : x : doubleVision xs
doubleVision [] = []
vision_test = doubleVision [1,2,3] ~=? [1,1,2,2,3,3]
\end{code}

\item<3-> Guards enable more flexible checking in pattern matching
\begin{code}
abs' x
  | x >= 0 = x
  | x < 0 = -x
\end{code}
\end{itemize}

\end{frame}

% -------------------

\begin{frame}[fragile]{Interlude - QuickCheck}

\begin{itemize}
\item Quickcheck property tests
\begin{code}
abs_check :: Integer -> Bool
abs_check x = abs' x >= 0
\end{code}

\item<2->
\begin{minted}{haskell}
Prelude> :load haskell.lhs
[1 of 1] Compiling Intro        ( haskell.lhs, interpreted )
Ok, modules loaded: Intro.
*Intro> quickCheck mymap_check
+++ OK, passed 100 tests.
\end{minted}
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Syntax and Type Signatures}

\begin{itemize}
\item<1-> Let bindings
\begin{code}
isPalindrome x = let rev = reverse x in
  rev == x
palindrome_test = isPalindrome "abba" ~=? True
\end{code}

\item<2-> Where bindings
\begin{code}
fibonacci n = fibonacci' n 1 0 where
  fibonacci' s c p = if s <= 1 
                     then c 
                     else fibonacci' (s - 1) (c + p) c 
\end{code}
\end{itemize}

\end{frame}

% --------------------

\section{Recursion and Higher Order Functions}

% --------------------

\begin{frame}[fragile]{Syntax and Type Signatures}

\begin{itemize}
\item What's wrong with this definition?
\begin{code}
mehFibonacci n
  | n <= 2     = 1
  | otherwise = mehFibonacci (n-1) + mehFibonacci (n-2)
mehFibonacci_test = mehFibonacci 5 ~=? fibonacci 5
\end{code}

\item<2-> 
\begin{minted}{haskell}
*Intro> mehFibonacci 100000
*BARF* *stackoverflow* *die*
\end{minted}

\item<3-> Solution: tail calls
\begin{minted}{haskell}
fibonacci n = fibonacci' n 1 0 where
  fibonacci' s c p = if s <= 1 
                     then c 
                     else fibonacci' (s - 1) (c + p) c 
\end{minted}
\end{itemize}

\end{frame}

% -------------------

\begin{frame}[fragile]{Higher order functions}
Here's how you implement map
\begin{itemize}
\item
\begin{code}
myMap :: (a -> b) -> [a] -> [b]
myMap f (x:xs) = f x : myMap f xs
myMap f [] = []

myMap_check :: [Integer] -> Bool
myMap_check x = myMap (+1) x == map (+1) x
\end{code}

\end{itemize}

\end{frame}

% --------------------

\section{Laziness}

% --------------------

\begin{frame}[fragile]{Laziness}
\begin{itemize}
\item Counting to infinity
\begin{code}
infty = [1..]
\end{code}

\item<2-> Forcing the issue?
\begin{code}
finiteSlice = take 100 infty
\end{code}

\item<3-> Don't try to reduce me to normal form..
\begin{code}
take' n x = x `deepseq` take n x
loooop :: [Integer]
loooop = take' 10 [1..]
\end{code}
\end{itemize}

\end{frame}

% ---------------------

\begin{frame}[fragile]{Laziness}

\begin{itemize}
\item Space leaks, or ``Oh my god, it's full of stars.."
\begin{minted}{haskell}
foldl (+) 0 [1..10000000]
\end{minted}

\item<2-> This is pretty fast
\begin{minted}{haskell}
import Data.List
foldl' (+) 0 [1..10000000]
\end{minted}

\item<3-> Defining a strict(ish) fold using $!$
\begin{code}
foldl'' f start xs = run start xs where
  run !acc (x:xs) = run (f acc x) xs
  run !acc [] = acc
\end{code}
\end{itemize}

\end{frame}

% --------------------

\section{Typeclasses and (Inductive) Data Types}

% --------------------

\begin{frame}[fragile]{Typeclasses}

\begin{itemize}
\item Problem - Ad hoc polymorphism:
\begin{minted}{haskell}
isEqual :: ? a -> a -> Bool 
isEqual x y = ?
\end{minted}

\item<2-> Solution:
\begin{code}
class Equality a where
  eq :: a -> a -> Bool

isEqual :: (Equality a) => a -> a -> Bool
isEqual x y = x `eq` y
\end{code}
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Algebraic Data Types}

\begin{itemize}
\item Defining new data types as sums of products
\begin{code}
data Boolean = BTrue | BFalse deriving (Show)
data Option a = Some a | None deriving (Eq, Show)
data ConsList a = CNil | Cons a (ConsList a) deriving (Eq, Show)
data BinTree a = Empty | Node (BinTree a) a (BinTree a) 
               deriving (Show)
\end{code}

\item<2-> Using them
\begin{code}
mapConsList :: (a -> b) -> ConsList a -> ConsList b
mapConsList f (Cons x xs) = Cons (f x) (mapConsList f xs)
mapConsList f CNil = CNil

mapConsList_test = mapConsList (* 2)
                   (Cons 1 $ Cons 2 $ CNil) ~=? 
                   (Cons 2 $ Cons 4 $ CNil)
\end{code}
\end{itemize}

\end{frame}

% -------------------

\begin{frame}[fragile]{Defining instances}

\begin{itemize}
\item Custom instances
\begin{code}
instance Equality Boolean where
  eq BTrue BTrue = True 
  eq BFalse BFalse = True
  eq _ _ = False
  
instance_tests = [BTrue `eq` BTrue ~=? True,
                  BTrue `eq` BFalse ~=? False]
\end{code}

\item<2-> Nesting
\begin{code}
instance Equality a => Equality (ConsList a) where
  eq x y = isEq True x y where
    isEq True CNil CNil = True
    isEq True (Cons x xs) (Cons y ys) = isEq (x `eq` y) xs ys
    isEq _ _ _ = False

instance_tests' = [Cons BTrue CNil `eq` Cons BTrue CNil ~=? True,
                   Cons BTrue CNil `eq` CNil ~=? False]
\end{code}
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Records}
\begin{itemize}
\item Defining a carrot
\begin{code}
data Carrot = Carrot { len :: Integer
                     , color :: String
                     , taste :: String
                     } deriving (Show, Eq)
some_carrot = Carrot { len = 8, 
                       color = "reddish", 
                       taste = "hints of carrot" }
newtype Color = Color { getColor :: String }
\end{code}

\item<2-> Accessors are auto generated
\begin{minted}{haskell}
*Intro> :t color
color :: Carrot -> String
\end{minted}

\item<3-> Most obnoxious thing in all of Haskell: record names must be unique within a given module
\end{itemize}

\end{frame}

% --------------------

\section{Functors, Idioms, Monads .. also Monoids}

% --------------------

\begin{frame}[fragile]{Monoids}

\begin{itemize}
\item Motivation: You want to combine some things, like strings
\begin{code}
combineStrings xs = foldl (++) "" xs 
\end{code}
But suddenly you have to combine (Option)al strings and would prefer not to repeat yourself

\item<2->
Monoids to the rescue
\begin{code}
class Monoid a where
  mempty :: a
  mappend :: a -> a -> a
  mconcat :: [a] -> a
  mconcat = foldr mappend mempty
\end{code}
\end{itemize}

\end{frame}

% -----------------

\begin{frame}[fragile]{Monoids}

\begin{itemize}
\item Instantiating..
\begin{code}
instance Monoid a => Monoid (Option a) where
  mempty = None
  mappend (Some x) (Some y) = Some (x `mappend` y)
  mappend None x = x
  mappend x None = x

-- strings are lists of characters
instance Monoid [a] where
  mempty = []
  mappend = (++)
\end{code}
\end{itemize}

\end{frame}

% -----------------

\begin{frame}[fragile]{Monoids}

\begin{itemize}
\item
\begin{code}
combineThings :: Monoid a => [a] -> a
combineThings xs = mconcat xs
                   
combining_tests = [
  combineThings ["foo", "bar", "baz"] ~=? 
  "foobarbaz",
  combineThings [Some "foo", None, Some "bar"] ~=?
  Some "foobar"]
\end{code}
\end{itemize}

\end{frame}

% -----------------

\begin{frame}[fragile]{Functors}

\begin{itemize}
\item Classic problem:
\begin{code}
getFromMaps x y =
  case lookup "foo" x of
    Just val -> case lookup "bar" y of
      Just val2 -> Just (val ++ val2)
      Nothing -> Nothing
    Nothing -> Nothing
-- ARGH
\end{code}

\item<2-> Or even just:
\begin{code}
doSomethingMaybe x = case x of
  Just a -> Just $ a ++ "bar"
  Nothing -> Nothing
\end{code}
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Functors}

\begin{itemize}
\item Enter functors
\begin{code}
doSomethingMaybe' x = fmap (++ "bar") x
\end{code}
\begin{minted}{haskell}
class Functor f where
  fmap :: (a -> b) -> f a -> f b
\end{minted}
\begin{code}
instance Functor Option where
  fmap f (Some x) = Some (f x)
  fmap f None = None
\end{code}

\item<2-> Option has \textit{kind} $* \rightarrow *$, which means it forms a concrete type of kind $*$ only when applied to a type argment.

\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Idioms - Applicative Functors}

\begin{itemize}
\item Queue Strauss' Sunrise - What if I could just write this
\begin{minted}{haskell}
getFromMaps' x y = (++) <$> lookup "foo" x <*> lookup "bar" y
\end{minted}
\item<2-> Applicative functors
\begin{code}
class (Functor f) => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

instance Applicative Option where
  pure = Some
  None <*> _ = None
  Some f <*> v = fmap f v
  
(<$>) :: (Functor f) => (a -> b) -> f a -> f b  
f <$> x = fmap f x  
\end{code}
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Monads}

\begin{itemize}
\item Even though they have many, many other uses, let's look at I/O. Idea:
\begin{minted}{haskell}
type IO a = RealWorld -> (a, RealWorld)
\end{minted}
\item<2-> Applicatives can't guarantee ordering of effects, we have to make sure that
things are evaluated and bound in the right sequence.
\item<3-> The IO Monad allows one to write
\begin{code}
getInputAndPrint :: IO ()
getInputAndPrint = do
  putStrLn "Give me some input"
  input <- getLine
  putStrLn input
\end{code}
making sure that the actions are performed in the specified sequence
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Monads}

\begin{itemize}
\item So how does this work? Let's start with a definition:
\begin{minted}{haskell}
class (Applicative m) => Monad m where
  return :: a -> m a
  (>>=) :: m a -> (a -> m b) -> m b
-- alternatively join :: m (m a) -> m a
\end{minted}
\item<2-> Option instance
\begin{code}
instance Monad Option where
  return = Some
  Some x >>= f = f x
  None >>= f = None
\end{code}
\end{itemize}

\end{frame}

% -------------------

\begin{frame}[fragile]{Monads}

\begin{itemize}
\item How to use it?
% Sadly I can't define this here since I don't know how to hide Monad from Prelude
\begin{code}
getFromMaps'' x y = lookup "foo" x >>= 
                    (\val1 -> (lookup "bar" y >>= 
                               (\val2 -> return $ val1 ++ val2)))
\end{code}
\item<2-> Since this doesn't look so nice, introduce some sugar
\begin{code}
getFromMaps''' x y = do
  val1 <- lookup "foo" x
  val2 <- lookup "bar" y
  return $ val1 ++ val2
\end{code}
\item<3-> List is also a monad
\begin{minted}{haskell}
do 
  x <- [1,2,3]
  fs <- [(+1), (+2)]
  return $ fs x
[2,3,3,4,4,5]
\end{minted}
\end{itemize}

\end{frame}

% --------------------

\begin{frame}[fragile]{Monads}

\begin{itemize}
\item That was easy, so let's move on to State..
\begin{code}
newtype State' s a = State' { runState :: s -> (a, s) }
instance Monad (State' a) where
  return x = State' $ \s -> (x, s)
  oldState >>= f = State' $ \s -> 
    let (intermediateVal, intermediateState) = runState oldState s
    in runState (f intermediateVal) intermediateState
\end{code}
\item<2-> Utilities
\begin{code}
put :: a -> State' a ()
put s = State' $ \_ -> ((), s)

get = State' $ \s -> (s, s)
\end{code}
\end{itemize}

\end{frame}

% --------------------

\section{Codata}

% --------------------

\begin{frame}[fragile]{Codata}

\begin{itemize}
\item $[1..]$ rewritten
\begin{code}
infty' = go 1 where
  go n = n : go (n + 1)
\end{code}
\item<2-> An infinite list of fibonacci numbers
\begin{code}
fibs = 1 : 1 : rec fibs where 
  rec (x:y:xs) = (x + y) : rec (y : xs)
cofibonacci n = head $ drop (n - 1) $ take n fibs 
\end{code}
\item<3-> The blurred distinction between data and codata
\begin{minted}{haskell}
Prelude> head $ 1 : undefined
1
\end{minted}
\end{itemize}

\end{frame}

% ---------------------

\section{Existential Types}

% ---------------------

\begin{frame}[fragile]{Existential Types}

\begin{minted}{haskell}
data Color' = Color' { red :: Integer, green :: Integer,
                       blue :: Integer, alpha :: Maybe Integer }
              deriving (Show, Eq)

class Colorizable a where
  colorize :: a -> Maybe Color'
-- I have omitted the instance declarations, insert here

data Colorlike = forall a. Colorizable a => Colorish

colorlist :: [Colorlike]
colorlist = [Colorish "255/50/0/20", Colorish [255, 50, 0, 20]]
\end{minted}

\end{frame}

% --------------------

\begin{frame}[fragile]{Epilogue}
\begin{center}
Join the Haskell meetup!: \url{http://www.meetup.com/NY-Haskell/}
\end{center}
\end{frame}
  
% --------------------

\begin{frame}[fragile]{Epilogue}
\begin{itemize}
\item Get the Haskell Platform at \url{http://www.haskell.org/platform/}
\item<2-> Install QuickCheck and HUnit using \mint{bash}|cabal install HUnit QuickCheck|
\item<3-> Grab the slides and run \mint{bash}|make extract| to produce plain Haskell code or \mint{bash}|runhaskell haskell.lhs| to just run it
\item<4-> To play around with it in ghci use \mint{haskell}|Prelude>:load haskell.lhs|
\end{itemize}
\end{frame}
  
% --------------------

\begin{frame}[fragile]{Epilogue}
\begin{code}
tests = test $ [map_test, map_test', vision_test,
                palindrome_test, mehFibonacci_test,
                mapConsList_test] ++ 
        ctest ++ instance_tests ++ instance_tests' ++
        combining_tests
main = do
  runTestTT tests
  quickCheck myMap_check
  quickCheck abs_check
\end{code}
\end{frame}

\end{document}
