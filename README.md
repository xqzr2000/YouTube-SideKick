# YouTube-SideKick

A Chrome side-panel extension that cleans up YouTube's recommendations for the video you're watching.

Inspired by Andrej Karpathy's [HackerTube project](https://karpathy.bearblog.dev/hacker-tube/), which replaces YouTube's recommendation feed with videos that Hacker News commenters actually recommended.

## Overview

Select text on a YouTube page, right-click, and choose **"SideKick's Selection"** to open the side panel. A local Python backend (running on Linux) handles the processing.

On the surface, SideKick is a browser extension. At its core, it's a data science project: I'm building a recommendation engine from scratch.
