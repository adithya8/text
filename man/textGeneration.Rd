% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/5_2_textGeneration.R
\name{textGeneration}
\alias{textGeneration}
\title{Text generation}
\usage{
textGeneration(
  x,
  model = "gpt2",
  device = "cpu",
  tokenizer_parallelism = FALSE,
  max_length = NULL,
  max_new_tokens = 20,
  min_length = 0,
  min_new_tokens = NULL,
  logging_level = "warning",
  force_return_results = FALSE,
  return_tensors = FALSE,
  return_full_text = TRUE,
  clean_up_tokenization_spaces = FALSE,
  prefix = "",
  handle_long_generation = NULL,
  set_seed = 202208L
)
}
\arguments{
\item{x}{(string)  A variable or a tibble/dataframe with at least one character variable.}

\item{model}{(string)  Specification of a pre-trained language model that have been trained with an
autoregressive language modeling objective, which includes the uni-directional models (e.g., gpt2).}

\item{device}{(string)  Device to use: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number}

\item{tokenizer_parallelism}{(boolean)  If TRUE this will turn on tokenizer parallelism.}

\item{max_length}{(Integer)  The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set. Defaults to NULL.}

\item{max_new_tokens}{(Integer)  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. The default value is 20.}

\item{min_length}{(Integer)  The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set. The default value is 0.}

\item{min_new_tokens}{(Integer)  The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt. Default is NULL.}

\item{logging_level}{(string)  Set the logging level.
Options (ordered from less logging to more logging): critical, error, warning, info, debug}

\item{force_return_results}{(boolean)  Stop returning some incorrectly formatted/structured results.
This setting does CANOT evaluate the actual results (whether or not they make sense, exist, etc.).
All it does is to ensure the returned results are formatted correctly (e.g., does the question-answering
dictionary contain the key "answer", is sentiments from textClassify containing the labels "positive"
 and "negative").}

\item{return_tensors}{(boolean)  Whether or not the output should include the prediction tensors (as token indices).}

\item{return_full_text}{(boolean) If FALSE only the added text is returned, otherwise the full text is returned.
(This setting is only meaningful if return_text is set to TRUE)}

\item{clean_up_tokenization_spaces}{(boolean)  Option to clean up the potential extra spaces in the returned text.}

\item{prefix}{(string) Option to add a prefix to prompt.}

\item{handle_long_generation}{By default, this function does not handle long generation
(those that exceed the model maximum length).}

\item{set_seed}{(Integer) Set seed.
(more info :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227).
This setting provides some ways to work around the problem:
None: default way, where no particular strategy is applied.
"hole": Truncates left of input, and leaves a gap that is wide enough to let generation happen.
(this might truncate a lot of the prompt and not suitable when generation exceed the model capacity)}
}
\value{
A tibble with generated text.
}
\description{
textGeneration() predicts the words that will follow a specified text prompt. (experimental)
}
\examples{
\donttest{
# generated_text <- textGeneration("The meaning of life is")
# generated_text
}
}
\seealso{
see \code{\link{textClassify}}, \code{\link{textNER}},
 \code{\link{textSum}}, \code{\link{textQA}}, \code{\link{textTranslate}}
}
