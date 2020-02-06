from rest_framework import serializers

class CommentSerializer(serializers.Serializer):
    url = serializers.CharField(max_length=200)
    vector = serializers.CharField(max_length=200)
    created = serializers.DateTimeField()