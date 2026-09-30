---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Progressive Probabilistic Circuits

A progressive probabilistic circuit (PPC) learns several tasks one after another without
forgetting the earlier ones. It transfers the idea of progressive neural networks
{cite}`rusu2016progressive` to probabilistic circuits: every task gets a circuit of its
own, called a column, and a new column may reuse what the earlier columns learned while
the earlier columns stay unchanged.

This page explains the structure of a PPC, how its columns are learned, and walks
through a small example with two tasks.

## Structure

A PPC is built from a *template*, a probabilistic circuit that fixes the structure every
column copies. Adding a column for a new task

1. copies the template into the PPC,
2. connects the new column to every earlier column: every sum unit of the new column
   receives the sum unit at the same position of each earlier column as an additional
   child, and
3. adds the root of the new column below the root of the PPC, a sum unit that mixes all
   columns.

Edges only ever point from a newer column to an older one, so an earlier column never
depends on a later one. Two aligned sum units model the same variables, so every sum
unit of a PPC stays smooth and every product unit stays decomposable. Determinism is not
kept: a sum unit's own children and the aligned units of earlier columns can model the
same rows.

The columns must have the same structure for the alignment to exist. If two columns
differ at an aligned position, adding the column raises one of the subclasses of
{py:class}`~probabilistic_model.exceptions.ColumnsDivergedError`.

## Learning

{py:class}`~probabilistic_model.learning.progressive.ProgressiveExpectationMaximization`
learns one column at a time with expectation maximization:

- While a column is learned, the root of the PPC gives it the whole weight. The column is
  therefore trained as the only model of its task. It can still use the earlier columns,
  because responsibilities flow into them through the edges the column has to them.
- The maximization step only updates the sum weights and leaf parameters of the learned
  column. Every other column keeps its parameters, which is what prevents forgetting.
- After learning, every column below the root is weighted by its share of all rows the
  columns were learned from.

Gaussian, discrete and symbolic leaves can be learned. Any other leaf raises an
{py:class}`~probabilistic_model.exceptions.UnsupportedLeafDistributionError`.

```{warning}
A later column reads from the earlier ones. Learning an earlier column again therefore
also changes what the later columns compute.
```
