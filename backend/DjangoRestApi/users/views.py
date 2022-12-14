from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework.parsers import JSONParser 
from rest_framework import status, response
from users.models import User
from users.serializers import UserSerializer , LoginSerializer ,RegistrationSerializer
from django.contrib.auth.hashers import check_password
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes


# def login(request):
#     if request.method == 'GET':
#         users = User.objects.all()
        
#         title = request.query_params.get('title', None)
#         if title is not None:
#             users = users.filter(title__icontains=title)
        
#         users_serializer = UserSerializer(users, many=True)
#         return JsonResponse(users_serializer.data, safe=False)
@api_view(['POST'])
def login(request):
    data=request.data
    serializers=LoginSerializer(data=data)
    if serializers.is_valid():
        user = User.objects.get(email=data['email'])
        print(user.password,user.check_password(data['password']))
        # user=authenticate(email=data['email'], password=data['password'])
        if user and user.check_password(data['password']):
            
            Msg={"message":"Login Successfull"}
            return response.Response(Msg,status=status.HTTP_200_OK)
        else: 
            return response.Response({"message": "Sorry, email or password not matched"},status=status.HTTP_400_BAD_REQUEST)
            
    else:
        Msg={"error":serializers.errors }
        return response.Response(Msg,status=status.HTTP_400_BAD_REQUEST)
        

@api_view(['POST'])
def register(request):
    print(request.data)
    serializer=RegistrationSerializer(data= request.data) 
    if serializer.is_valid():
        serializer.save()
        
        Msg={"message":"Register Successfull"}
        return response.Response(Msg,status=status.HTTP_200_OK)
    else:
        Msg={"error":serializer.errors }
        return response.Response(Msg,status=status.HTTP_400_BAD_REQUEST)



    




@api_view(['GET', 'POST', 'DELETE'])
def user_list(request):
    if request.method == 'GET':
        try:
            users = User.objects.all()
                        
            title = request.query_params.get('title', None)
            if title is not None:
                users = users.filter(title__icontains=title)
            
            users_serializer = UserSerializer(users, many=True)
            return JsonResponse(users_serializer.data, safe=False)
        except: 
            return JsonResponse({'message': 'The users does not exist'}, status=status.HTTP_404_NOT_FOUND)
    
    elif request.method == 'POST':
        user_data = JSONParser().parse(request)
        user_serializer = UserSerializer(data=user_data)
        if user_serializer.is_valid():
            user_serializer.save()
            return JsonResponse(user_serializer.data, status=status.HTTP_201_CREATED) 
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        count = User.objects.all().delete()
        return JsonResponse({'message': '{} Users were deleted successfully!'.format(count[0])}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'PUT', 'DELETE'])
def user_detail(request, pk):
    try: 
        user = User.objects.get(pk=pk) 
    except User.DoesNotExist: 
        return JsonResponse({'message': 'The user does not exist'}, status=status.HTTP_404_NOT_FOUND) 
 
    if request.method == 'GET': 
        user_serializer = UserSerializer(user) 
        return JsonResponse(user_serializer.data) 
 
    elif request.method == 'PUT': 
        user_data = JSONParser().parse(request) 
        user_serializer = UserSerializer(user, data=user_data) 
        if user_serializer.is_valid(): 
            user_serializer.save() 
            return JsonResponse(user_serializer.data) 
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
 
    elif request.method == 'DELETE': 
        user.delete() 
        return JsonResponse({'message': 'User was deleted successfully!'}, status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def get_detail(request):
    return JsonResponse({
        'id': request.user.id,
        'is_approved': request.user.is_approved,
        'contract_signed': request.user.contract_signed,
        'username': request.user.username,
        'email': request.user.email
    })
