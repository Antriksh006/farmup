package com.farmup.farmup_java_backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class OtpService {

    @Value("${fast2sms.api.key}")
    private String apiKey;

    @Value("${fast2sms.api.url}")
    private String apiUrl;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ConcurrentHashMap<String, String> otpStore = new ConcurrentHashMap<>();

    public void sendOtp(String phoneNumber) {
        String otp = String.valueOf(new Random().nextInt(900000) + 100000);

        String message = "Your OTP for login is " + otp;

        HttpHeaders headers = new HttpHeaders();
        headers.set("authorization", apiKey);
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        String body = "route=v3&sender_id=TXTIND&message=" + message +
                "&language=english&flash=0&numbers=" + phoneNumber;

        HttpEntity<String> entity = new HttpEntity<>(body, headers);

        restTemplate.exchange(apiUrl, HttpMethod.POST, entity, String.class);

        otpStore.put(phoneNumber, otp);
    }

    public boolean verifyOtp(String phoneNumber, String otp) {
        return otp.equals(otpStore.get(phoneNumber));
    }
}