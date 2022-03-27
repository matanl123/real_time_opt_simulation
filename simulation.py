import numpy as np
import heapq

np.random.seed(0)
## next node of vehicle - (id, next_node, time_at_next_node)
## k = len(vehicle) מימד המערך

## next_node - x ,y

## clock

## default insertion - vehicle with munimum time to arrive to the new request
##
class node():
    x,
    y

class vehicle():
    __init__(self, id, idle = True, idle_node  = node(0,0)):
        id,
        list_of_next_stops = []
        idle
        idle_node

class event:
    time,
    event_type (request_arrival, vehicle_departure(update pickup_time/delivery_time)/ delete first elemnt from list_of_next_stops) ## each time requst arrival
    vehicle_id

def __init__(time, event_type,vehicle_id=-1):


    def __lt__(self, other):
        return self.time < other.time

    def __le__(self, other):
        return self.time <= other.time


class request():
    id,
    node_pickup node,
    node_delivery node,
    request_time,
    pickup_time,
    delivery_time

class stop():
    stop_type ##pickup/delivery
    request_id,
    departure_time,

parcel_arrival_event  = 0
vehicle_departure_event = 1

t = 0
T = 10
lamb = 1
events = []
requests = {}
vehicles = {}
requests_count = 0
heapq.heappush(events, (event(np.random.exponential(lamb), parcel_arrival_event)))

def insert_new_request(request_id):
    ## update requests_count with new request
    #  insert first event if the vehicle was idle



def generate_nodes():
    return node(np.random.rand(0,10), np.random.rand(0,10))

def dist(node1, node2):
    # distance between two nodes

while t < T:

    curr_event = heapq.heappop(events)
    t = curr_event.time
    if curr_event.event_type == parcel_arrival_event:
        heapq.heappush(events, (t + event(np.random.exponential(lamb), parcel_arrival_event)))
        node_pickup  = genrate_nodes()
        node_delivery = genrate_nodes()
        requests[requests_count] = request(requests_count, node_pickup, node_delivery, t, None, None)
        insert_new_request(requests_count)
        requests_count +=1
    elif curr_event.event_type == vehicle_departure_event:
        # delete first request from vehicle
        # insert new vehicle departure event unless the list is empty







