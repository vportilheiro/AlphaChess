# AlphaChess
This project attemtps to take the approach of AlphaGo, and apply it to learning the game of chess.
It was started as the final project for Stanford's CS221 AI course.

The relevant Google Deep Mind paper is available at: http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html.

As a general overview, the program is trained in two steps, a supervised learning step where a policy network learns to imitate human experts,
and a self-play reinforcement learning step.

The course project tackled the first step, with moderate success, applying certain techniques and comparing results
first described Barak Oshri and Nishith Khandwala (http://cs231n.stanford.edu/reports/ConvChess.pdf).
