Repo for the ACL 2022 short paper “Counterfactual Explanations for Natural Language Interfaces” by George Tolkachev, Stephen Mell, Steve Zdancewic, and Osbert Bastani.

Paper link: https://arxiv.org/abs/2204.13192

Video presentation: https://www.youtube.com/watch?v=dwl1pWxo3ho

Our contribution is a novel algorithm for computing counterfactual explanations for semantic parsers. We assume the following process:

1. User provides a command, but the NL interface fails to generate the desired result
2. User now provides desired result (in the form of a denotation/trajectory to goal) in addition to command
3. Our algorithm computes an alternative utterance (counterfactual explanation) that:

   a. Is as close as possible to the original utterance
   
   b. The semantic parser correctly processes (i.e. user achieves desired result)

The outcome of this process is that the user has a better understanding of how to modify commands to achieve their goals in future interactions with the system.

Here is an illustration of the above process:

![alt text](https://github.com/georgeto20/counterfactual_explanations/blob/main/process.png?raw=true "Illustration of Process")
