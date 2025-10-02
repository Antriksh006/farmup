package com.farmup.farmup_java_backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class EmailService {

    @Value("${brevo.api-key}")
    private String brevoApiKey;

    @Value("${brevo.sender.email}")
    private String senderEmail;

    @Value("${brevo.sender.name}")
    private String senderName;

    private final RestTemplate restTemplate = new RestTemplate();

    private static final String BREVO_EMAIL_API_URL = "https://api.brevo.com/v3/smtp/email";

    public void sendOtpEmail(String toEmail, String otp) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("api-key", brevoApiKey);

        Map<String, Object> sender = Map.of(
                "name", senderName,
                "email", senderEmail
        );

        Map<String, Object> recipient = Map.of(
                "email", toEmail
        );

        String subject = "Your FarmUp OTP";
        String htmlContent = "<p>Hi there,</p>"
                + "<p>Your One-Time Password (OTP) for signup is: "
                + "<strong style='font-size: 18px;'>" + otp + "</strong></p>"
                + "<p>This OTP is valid for 10 minutes.</p>"
                + "<br><p>Thanks,<br>Team FarmUp</p>";

        Map<String, Object> body = new HashMap<>();
        body.put("sender", sender);
        body.put("to", List.of(recipient));
        body.put("subject", subject);
        body.put("htmlContent", htmlContent);

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(BREVO_EMAIL_API_URL, request, String.class);
            System.out.println("Email sent: " + response.getStatusCode());
        } catch (Exception e) {
            System.err.println("Failed to send email: " + e.getMessage());
        }
    }

    public void sendResetOtpEmail(String toEmail, String otp) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("api-key", brevoApiKey);

        Map<String, Object> sender = Map.of(
                "name", senderName,
                "email", senderEmail
        );

        Map<String, Object> recipient = Map.of(
                "email", toEmail
        );

        String subject = "FarmUp Password Reset OTP";
        String htmlContent = "<p>Hello,</p>"
                + "<p>You requested to reset your password. Please use the OTP below to proceed:</p>"
                + "<p style='font-size: 20px; font-weight: bold;'>" + otp + "</p>"
                + "<p>This OTP is valid for 10 minutes. If you didn't request a password reset, you can safely ignore this message.</p>"
                + "<br><p>– Team FarmUp</p>";

        Map<String, Object> body = new HashMap<>();
        body.put("sender", sender);
        body.put("to", List.of(recipient));
        body.put("subject", subject);
        body.put("htmlContent", htmlContent);

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(BREVO_EMAIL_API_URL, request, String.class);
            System.out.println("Reset OTP email sent: " + response.getStatusCode());
        } catch (Exception e) {
            System.err.println("Failed to send reset OTP email: " + e.getMessage());
        }
    }

}