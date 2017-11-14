## greet
* _greet
    - utter_greet

## happy
* _thankyou
    - utter_youarewelcome

## goodbye
* _goodbye
    - utter_goodbye

## venue_search
* _search_venues
    - action_search_venues
    - slot{"venues": [{"name": "Big Arena", "reviews": 4.5}]}

## concert_search
* _search_concerts
    - action_search_concerts
    - slot{"concerts": [{"artist": "Foo Fighters", "reviews": 4.5}]}

## compare_reviews_venues
* _compare_reviews
    - action_show_venue_reviews

## compare_reviews_concerts
* _compare_reviews
    - action_show_concert_reviews

## Generated Story -5015531891657488253
* _greet
    - utter_greet
* _search_concerts
    - action_search_concerts
    - slot{"concerts": [{"artist": "Foo Fighters", "reviews": 4.5}, {"artist": "Katy Perry", "reviews": 5.0}]}
* _thankyou
    - utter_youarewelcome
* _goodbye
    - utter_goodbye
    - export
