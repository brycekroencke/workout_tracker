# **Workout Tracker Notes**

## **DATA PROCESSING**

#### Movement Classifier and Bar Tracking

- ###### Current
  - process frames of webcam / vid source
  - track user face (removed)
  - track bar movement and trace line
  - rep and set counter

- ###### Future
  - create MVP app
    - v1 upload video and preprocess data (?? + Python on server side)
      - [flask](https://flask.palletsprojects.com/en/1.1.x/) server backend
      - [progressive web app](https://web.dev/progressive-web-apps/)
      - [svelte](https://svelte.dev/) framework for web apps
  - expand models domain of applicability by growing the dataset
    - record personal training sessions
    - clip YouTube training videos
  - find reliable tracking point

#### Form Critique and Data Analysis
- ###### Current
  - N/A
- ###### Future
  - judge verticality of bar movement
  - graph users lifting activity and progression
  - reincorporate Posenet??

#### Weight Detection
- ###### Current
  - N/A
- ###### Future
  - user input after a set (currently most achievable)
  - sticker for weight values
  - voice recognition to convey weight
  - something at end of bar to communicate current weight

## **User Experience**

#### Social Aspects
- ###### Current
  - N/A
- ###### Future
  - share and see friends lifting data
  - provide some method to challenge / compete with friends
