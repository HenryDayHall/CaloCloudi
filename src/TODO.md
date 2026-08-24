# ASAP
- Make the pcfm for multiple particles
    * wire this up to the summary_plots_best - currently waiting on the summaries to finish, then I will have a check for this.
- Summarise against other models.
    * some kind of summary stat - sliced wasserstein
    * timing - need to check the timing of the compiled model.


# at some point in the future
- Why does the EM data not load?
- Update the unit tests to meet the changes
- write a lot of unit tests (maybe claude can do that)
- Go through the unintended features that claude's unit tests turned up and fix or document
- Add comments that explain things to the config files from the old config files
- Consider making a subdictionary for the diffusion parts of the model
- Other models - AS - CC3
- Have a look at Mose on the PointCountFM_private, see if it can also do something
- More elegant solution for energy units correction.

# Slides planned
- Updates and Summary of CaloClouds 3.5
    * No plots, just a diagram
- Motivation for updates
    * Steal plots from last presentation
- New particles available
    * A performance plot comparing generalist w/ only photons to generalist w/ e+e-and photons
    ^ Done - Remember to only compare photon data....
- Summary of experiment with MoE and MoS, note that num parameters held constant
    * Performance plots comparing 3 models, for just one part of CaloClouds
    ^ TODO
- Timing with MoE and MoS
    * Can isolate the part that's changed and just time that
    ^ TODO

# extras for poster
I don't strictly need these, I can make a poster just based on the slides
- Comparison to other models would be nice
- Summary stats would be nice
- Other component MoE and MoS would be nice
