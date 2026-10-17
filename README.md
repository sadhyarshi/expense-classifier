# Expense Classifier

A tiny Naive Bayes text classifier written from scratch in Java. No libraries, no build tool.

It reads an expense or SMS text and predicts a category such as food, fuel, client payment, tools, bills or travel.

## Requirements

Java (JDK) 11 or newer.

## Run

**IntelliJ IDEA:** open the folder as a project, open `ExpenseClassifier.java`, and click the green Run button next to `main`. Then click inside the Run console at the bottom, type a line and press Enter.

**Terminal:** from this folder, run:

    java ExpenseClassifier.java

## Use

Type an expense or SMS text at the `>` prompt and press Enter. It prints the best category and the second-best guess.

    > Rs 450 debited at HP petrol pump
      -> fuel (85%)
         next: tools (5%)

### Teach it

Start the line with `+`, then the category, a colon, and some example text. A new category name creates a new category.

    > + medical: Apollo pharmacy medicines
      learned.

The `+` at the start is required. Without it, the line is treated as text to classify, not as training.

Teach a few examples per category for better results:

    + medical: doctor consultation fee clinic
    + medical: hospital bill lab test

### Reading the confidence

A high percentage means the model is fairly sure. Low or tied percentages (for example 19% and 19%) mean it has not seen those words before and is guessing. Teach it more examples for that category.

### Quit

Type `exit`.

## Limits

- What you teach it is lost when the program quits. Nothing is saved to disk yet.
- It only learns from words (letters). Numbers and amounts are ignored.
- Spelling matters: `petrol` and `petrl` are different words to it.

## Customise

Edit the `seed()` method in `ExpenseClassifier.java` to change the built-in categories and training examples. Examples added there are always loaded at startup.
