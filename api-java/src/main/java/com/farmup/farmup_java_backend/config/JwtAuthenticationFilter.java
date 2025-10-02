package com.farmup.farmup_java_backend.config;

import com.farmup.farmup_java_backend.model.User;
import com.farmup.farmup_java_backend.repository.UserRepository;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    private final JwtTokenProvider jwtUtil;
    private final UserRepository userRepository;

    public JwtAuthenticationFilter(JwtTokenProvider jwtUtil, UserRepository userRepository) {
        this.jwtUtil = jwtUtil;
        this.userRepository = userRepository;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        System.out.println("🔥 JwtAuthenticationFilter is running for request: " + request.getRequestURI());

        String authHeader = request.getHeader("Authorization");
        String token = null;
        String username = null;

        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            token = authHeader.substring(7);
            username = jwtUtil.extractPhone(token);
        }
        System.out.println("Extracted username from token (email): " + username);


        if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {

            User user = userRepository.getByPhone(username);
            if (user == null) {
                return;
            }
            System.out.println("Resolved userId from DB: " + user.getId());

            List<SimpleGrantedAuthority> authorities = List.of(
                    new SimpleGrantedAuthority("ROLE_USER")
            );
            UserDetails userDetails = org.springframework.security.core.userdetails.User
                    .withUsername(user.getPhone())
                    .password("") // Empty password since we're using JWT
                    .authorities(authorities)
                    .build();

            if (jwtUtil.isTokenValid(token, userDetails)) {
                UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(
                        userDetails, null, userDetails.getAuthorities());
                authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                SecurityContextHolder.getContext().setAuthentication(authToken);
                System.out.println("✅ Set authentication for userId: " + user.getId());
            }



        }
        filterChain.doFilter(request, response);
    }
}
